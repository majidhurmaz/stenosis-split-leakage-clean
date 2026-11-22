"""Custom AST detectors for Python cryptographic misuse.

Bandit and Semgrep miss several crypto-specific problems (low PBKDF2 iterations, static IVs,
fast hashes for passwords, non-constant-time comparison of MACs). These detectors fill the gap.
Every label maps to one CWE; `severity` is "misuse" (counted in the primary outcome) or
"weak" (reported separately, not counted).

Thresholds follow the OWASP Password Storage Cheat Sheet (PBKDF2-HMAC-SHA256 >= 600,000;
PBKDF2-HMAC-SHA512 >= 210,000) and NIST SP 800-131A (RSA >= 2048 bits). Cite the version
you checked in the paper.
"""
from __future__ import annotations

import ast
import re

LABELS = {
    # label: (CWE, severity, short description)
    "WEAK_HASH":          ("CWE-328", "misuse", "MD5/SHA-1 used for a security purpose"),
    "FAST_PASSWORD_HASH": ("CWE-916", "misuse", "Password hashed with a fast hash, no KDF"),
    "LOW_KDF_ITERATIONS": ("CWE-916", "misuse", "PBKDF2 iterations below OWASP minimum"),
    "STATIC_SALT":        ("CWE-760", "misuse", "Constant salt"),
    "WEAK_CIPHER":        ("CWE-327", "misuse", "DES/3DES/RC4/RC2/Blowfish"),
    "ECB_MODE":           ("CWE-327", "misuse", "Block cipher in ECB mode"),
    "STATIC_IV":          ("CWE-329", "misuse", "Constant IV or nonce"),
    "HARDCODED_KEY":      ("CWE-321", "misuse", "Hard-coded key or secret"),
    "WEAK_RANDOM":        ("CWE-338", "misuse", "random module used for secrets"),
    "SHORT_RSA_KEY":      ("CWE-326", "misuse", "RSA/DSA key shorter than 2048 bits"),
    "TLS_NO_VERIFY":      ("CWE-295", "misuse", "Certificate or hostname verification disabled"),
    "OLD_TLS":            ("CWE-327", "misuse", "SSLv2/v3 or TLS 1.0/1.1 allowed"),
    "JWT_NO_VERIFY":      ("CWE-347", "misuse", "JWT signature not verified or alg 'none'"),
    "TIMING_COMPARE":     ("CWE-208", "misuse", "MAC/token compared with == instead of compare_digest"),
    "UNAUTH_ENCRYPTION":  ("CWE-353", "weak",   "CBC/CTR encryption without a MAC"),
    "PRIVKEY_UNENCRYPTED": ("CWE-312", "weak",  "Private key written without encryption"),
    "RSA_PKCS1V15_ENC":   ("CWE-780", "weak",   "RSA encryption with PKCS#1 v1.5 padding"),
    "DEPRECATED_PYCRYPTO": ("CWE-1104", "weak", "Unmaintained pycrypto import (Crypto.*) - check if pycryptodome"),
    "HARDCODED_PASSWORD": ("CWE-259", "weak",   "Bandit B105-B107: literal password (often demo data)"),
}

SECRET_NAME = re.compile(r"(^|_)(key|secret|secret_key|api_key|aes_key|enc_key|encryption_key|hmac_key|"
                         r"jwt_secret|signing_key|private_key|passphrase|master_key)$", re.I)
IV_NAME = re.compile(r"(^|_)(iv|nonce|init_vector|initialization_vector)$", re.I)
SALT_NAME = re.compile(r"(^|_)salt$", re.I)
PW_NAME = re.compile(r"pass|pwd|pw\b|^pw|parol", re.I)
MAC_NAME = re.compile(r"sig|signature|hmac|digest|mac\b|_mac|token|otp|hash", re.I)
RANDOM_FUNCS = {"random", "randint", "randrange", "choice", "choices", "sample", "getrandbits", "shuffle", "uniform"}
WEAK_CIPHER_NAMES = {"DES", "DES3", "TripleDES", "ARC4", "ARC2", "Blowfish", "CAST", "IDEA", "RC4", "RC2"}
FAST_HASHES = {"md5", "sha1", "sha224", "sha256", "sha384", "sha512", "sha3_256", "sha3_512", "blake2b", "blake2s"}
KDF_HINT = re.compile(r"pbkdf2|bcrypt|argon2|scrypt|PasswordHasher|passlib|make_password|generate_password_hash|"
                      r"check_password_hash|hashpw", re.I)
OLD_TLS_ATTRS = {"PROTOCOL_SSLv2", "PROTOCOL_SSLv3", "PROTOCOL_TLSv1", "PROTOCOL_TLSv1_1", "SSLv3", "TLSv1", "TLSv1_1"}


def _name(node) -> str:
    """Dotted name of a Name/Attribute/Call func, e.g. 'hashlib.md5'."""
    if isinstance(node, ast.Call):
        node = node.func
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_const_bytes_or_str(node) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (bytes, str)):
        return len(node.value) >= 4
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):  # b"\x00" * 16
        return _is_const_bytes_or_str(node.left) or _is_const_bytes_or_str(node.right) or \
            (isinstance(node.left, ast.Constant) and isinstance(node.left.value, bytes))
    if isinstance(node, ast.Call) and _name(node).endswith("encode") and isinstance(node.func, ast.Attribute):
        return _is_const_bytes_or_str(node.func.value)
    if isinstance(node, ast.Call) and _name(node) in ("bytes", "bytearray", "bytes.fromhex") and node.args:
        return isinstance(node.args[0], (ast.Constant, ast.List))
    return False


def _int_const(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, int) else None


def _kw(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _target_names(t) -> list[str]:
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, ast.Attribute):
        return [t.attr]
    if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
        return [t.slice.value]  # app.config["SECRET_KEY"] = "..."
    if isinstance(t, (ast.Tuple, ast.List)):
        return [n for e in t.elts for n in _target_names(e)]
    return []


class _V(ast.NodeVisitor):
    def __init__(self, src: str):
        self.src = src
        self.hits: list[tuple[str, int, str]] = []
        self.has_compare_digest = "compare_digest" in src
        self.has_mac = bool(re.search(r"\bhmac\b|HMAC|AESGCM|ChaCha20Poly1305|MODE_GCM|MODE_EAX|MODE_CCM|MODE_SIV|"
                                      r"MODE_OCB|Fernet|modes\.GCM", src))
        self.has_kdf = bool(KDF_HINT.search(src))
        self.random_imported_as_module = bool(re.search(r"^\s*import\s+random\b", src, re.M))
        self.random_funcs_imported = set()
        for m in re.finditer(r"^\s*from\s+random\s+import\s+(.+)$", src, re.M):
            self.random_funcs_imported |= {x.strip().split(" as ")[-1] for x in m.group(1).split(",")}

    def hit(self, label, node, why=""):
        self.hits.append((label, getattr(node, "lineno", 0), why))

    # ---- imports
    def visit_ImportFrom(self, node):
        if node.module and (node.module == "Crypto" or node.module.startswith("Crypto.")):
            if "pycryptodome" not in self.src.lower():
                self.hit("DEPRECATED_PYCRYPTO", node, node.module)
            for a in node.names:
                if a.name in WEAK_CIPHER_NAMES:
                    self.hit("WEAK_CIPHER", node, a.name)
        self.generic_visit(node)

    # ---- calls
    def visit_Call(self, node):
        n = _name(node)
        last = n.split(".")[-1]

        # hashes
        if n in ("hashlib.md5", "hashlib.sha1", "md5", "sha1") or last in ("MD5", "SHA1") or \
                (n == "hashlib.new" and node.args and isinstance(node.args[0], ast.Constant)
                 and str(node.args[0].value).lower() in ("md5", "sha1")):
            ufs = _kw(node, "usedforsecurity")
            if not (isinstance(ufs, ast.Constant) and ufs.value is False):
                self.hit("WEAK_HASH", node, n)
        if (n.startswith("hashlib.") and last in FAST_HASHES) and not self.has_kdf:
            argsrc = " ".join(ast.unparse(a) for a in node.args)
            if PW_NAME.search(argsrc):
                self.hit("FAST_PASSWORD_HASH", node, argsrc[:60])

        # PBKDF2 iterations
        if last == "pbkdf2_hmac":
            it = node.args[3] if len(node.args) > 3 else _kw(node, "iterations")
            alg = node.args[0] if node.args else _kw(node, "hash_name")
            self._check_iter(node, it, alg)
            salt = node.args[2] if len(node.args) > 2 else _kw(node, "salt")
            if salt is not None and _is_const_bytes_or_str(salt):
                self.hit("STATIC_SALT", node)
        if last in ("PBKDF2HMAC", "PBKDF2"):
            it = _kw(node, "iterations") or _kw(node, "count")
            if last == "PBKDF2" and it is None and len(node.args) > 3:
                it = node.args[3]
            if last == "PBKDF2" and it is None:
                self.hit("LOW_KDF_ITERATIONS", node, "PBKDF2 default count (1000)")
            else:
                self._check_iter(node, it, _kw(node, "algorithm") or _kw(node, "hmac_hash_module"))
            salt = _kw(node, "salt") or (node.args[1] if last == "PBKDF2" and len(node.args) > 1 else None)
            if salt is not None and _is_const_bytes_or_str(salt):
                self.hit("STATIC_SALT", node)

        # ciphers and modes
        if last == "new" and n.split(".")[0] in WEAK_CIPHER_NAMES:
            self.hit("WEAK_CIPHER", node, n)
        if last in WEAK_CIPHER_NAMES and ("algorithms" in n or n == last):
            self.hit("WEAK_CIPHER", node, n)
        if last == "ECB" and "modes" in n:
            self.hit("ECB_MODE", node)
        if last in ("CBC", "CTR", "CFB", "OFB", "GCM") and "modes" in n and node.args:
            if _is_const_bytes_or_str(node.args[0]):
                self.hit("STATIC_IV", node, n)
        if last == "new" and n.endswith(("AES.new", "DES3.new", "DES.new", "ChaCha20.new")):
            for a in node.args[1:]:
                if isinstance(a, ast.Attribute) and a.attr == "MODE_ECB":
                    self.hit("ECB_MODE", node)
            for a in node.args[2:3]:
                if _is_const_bytes_or_str(a):
                    self.hit("STATIC_IV", node)
            for kw in ("iv", "nonce"):
                v = _kw(node, kw)
                if v is not None and _is_const_bytes_or_str(v):
                    self.hit("STATIC_IV", node, kw)
            if node.args and _is_const_bytes_or_str(node.args[0]):
                self.hit("HARDCODED_KEY", node, "literal key passed to cipher")
            if len(node.args) == 1 and _kw(node, "mode") is None and "AES.new" in n:
                self.hit("ECB_MODE", node, "AES.new without mode")  # PyCryptodome requires mode; old pycrypto defaulted ECB
        if last in ("AESGCM", "ChaCha20Poly1305", "Fernet", "AESCCM") and node.args and _is_const_bytes_or_str(node.args[0]):
            self.hit("HARDCODED_KEY", node, f"literal key to {last}")
        if last == "encrypt" and isinstance(node.func, ast.Attribute) and node.args:
            # AESGCM(key).encrypt(nonce, data, aad) with literal nonce
            if _name(node.func.value).split(".")[-1] in ("AESGCM", "ChaCha20Poly1305", "AESCCM") and \
                    _is_const_bytes_or_str(node.args[0]):
                self.hit("STATIC_IV", node, "literal AEAD nonce")
        if last == "new" and n.endswith("hmac.new") and node.args and _is_const_bytes_or_str(node.args[0]):
            self.hit("HARDCODED_KEY", node, "literal HMAC key")

        # randomness
        if (self.random_imported_as_module and n.startswith("random.") and last in RANDOM_FUNCS) or \
                (n in self.random_funcs_imported and n in RANDOM_FUNCS):
            self.hit("WEAK_RANDOM", node, n)
        if n == "uuid.uuid1":
            self.hit("WEAK_RANDOM", node, "uuid1 is time-based")

        # key sizes
        if last in ("generate_private_key", "generate") and ("rsa" in n.lower() or "dsa" in n.lower()):
            ks = _kw(node, "key_size") or _kw(node, "bits")
            if ks is None and n.endswith(("RSA.generate", "DSA.generate")) and node.args:
                ks = node.args[0]
            v = _int_const(ks) if ks is not None else None
            if v is not None and v < 2048:
                self.hit("SHORT_RSA_KEY", node, str(v))
        if n.endswith("rsa.newkeys") and node.args and (_int_const(node.args[0]) or 9999) < 2048:
            self.hit("SHORT_RSA_KEY", node, "rsa.newkeys")

        # TLS
        v = _kw(node, "verify")
        if v is not None and isinstance(v, ast.Constant) and v.value is False:
            self.hit("TLS_NO_VERIFY", node, "verify=False")
        if last in ("_create_unverified_context", "disable_warnings"):
            self.hit("TLS_NO_VERIFY", node, last)
        if last in ("SSLContext", "wrap_socket") and node.args and isinstance(node.args[0], ast.Attribute) \
                and node.args[0].attr in OLD_TLS_ATTRS:
            self.hit("OLD_TLS", node, node.args[0].attr)
        sv = _kw(node, "ssl_version")
        if isinstance(sv, ast.Attribute) and sv.attr in OLD_TLS_ATTRS:
            self.hit("OLD_TLS", node, sv.attr)
        cr = _kw(node, "cert_reqs")
        if isinstance(cr, ast.Attribute) and cr.attr == "CERT_NONE":
            self.hit("TLS_NO_VERIFY", node, "cert_reqs=CERT_NONE")

        # JWT
        if n.endswith("jwt.decode") or last == "decode" and "jwt" in n.lower():
            opts = _kw(node, "options")
            if opts is not None and re.search(r"verify_signature['\"]\s*:\s*False", ast.unparse(opts)):
                self.hit("JWT_NO_VERIFY", node, "verify_signature False")
            vk = _kw(node, "verify")
            if isinstance(vk, ast.Constant) and vk.value is False:
                self.hit("JWT_NO_VERIFY", node, "verify=False")
            algs = _kw(node, "algorithms")
            if algs is not None and "none" in ast.unparse(algs).lower():
                self.hit("JWT_NO_VERIFY", node, "alg none allowed")
            if len(node.args) >= 2 and _is_const_bytes_or_str(node.args[1]):
                self.hit("HARDCODED_KEY", node, "literal JWT secret")
        if (n.endswith("jwt.encode") or last == "encode" and "jwt" in n.lower()):
            alg = _kw(node, "algorithm")
            if isinstance(alg, ast.Constant) and str(alg.value).lower() == "none":
                self.hit("JWT_NO_VERIFY", node, "alg none")
            if len(node.args) >= 2 and _is_const_bytes_or_str(node.args[1]):
                self.hit("HARDCODED_KEY", node, "literal JWT secret")

        # weak practice
        if n.endswith("NoEncryption"):
            self.hit("PRIVKEY_UNENCRYPTED", node)
        if last == "PKCS1v15" and "padding" in n and "encrypt" in self.src:
            if re.search(r"\.encrypt\([^)]*PKCS1v15", self.src, re.S):
                self.hit("RSA_PKCS1V15_ENC", node)
        if n.endswith("PKCS1_v1_5.new") and "Cipher" in self.src and "Signature" not in n:
            if re.search(r"from\s+Crypto\.Cipher\s+import\s+[^\n]*PKCS1_v1_5", self.src):
                self.hit("RSA_PKCS1V15_ENC", node)
        self.generic_visit(node)

    def _check_iter(self, node, it, alg):
        v = _int_const(it) if it is not None else None
        if v is None:
            return  # variable: leave to manual review
        algs = ast.unparse(alg).lower() if alg is not None else "sha256"
        need = 210_000 if "512" in algs else 1_300_000 if "sha1" in algs else 600_000
        if v < need:
            self.hit("LOW_KDF_ITERATIONS", node, f"{v} < {need}")

    # ---- attributes / assignments
    def visit_Attribute(self, node):
        if node.attr == "MODE_ECB":
            self.hit("ECB_MODE", node)
        if node.attr in ("PROTOCOL_SSLv2", "PROTOCOL_SSLv3", "PROTOCOL_TLSv1", "PROTOCOL_TLSv1_1"):
            self.hit("OLD_TLS", node, node.attr)
        if node.attr in ("MODE_CBC", "MODE_CTR", "MODE_CFB", "MODE_OFB") and not self.has_mac:
            self.hit("UNAUTH_ENCRYPTION", node, node.attr)
        self.generic_visit(node)

    def visit_Assign(self, node):
        names = [x for t in node.targets for x in _target_names(t)]
        self._assign(names, node.value, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self._assign(_target_names(node.target), node.value, node)
        self.generic_visit(node)

    def _assign(self, names, value, node):
        for nm in names:
            if nm in ("check_hostname",) and isinstance(value, ast.Constant) and value.value is False:
                self.hit("TLS_NO_VERIFY", node, "check_hostname=False")
            if nm == "verify_mode" and isinstance(value, ast.Attribute) and value.attr == "CERT_NONE":
                self.hit("TLS_NO_VERIFY", node, "CERT_NONE")
            if nm == "minimum_version" and isinstance(value, ast.Attribute) and value.attr in OLD_TLS_ATTRS:
                self.hit("OLD_TLS", node, value.attr)
            if nm == "verify" and isinstance(value, ast.Constant) and value.value is False:
                self.hit("TLS_NO_VERIFY", node, "session.verify=False")
            if _is_const_bytes_or_str(value):
                if SECRET_NAME.search(nm):
                    self.hit("HARDCODED_KEY", node, nm)
                elif IV_NAME.search(nm):
                    self.hit("STATIC_IV", node, nm)
                elif SALT_NAME.search(nm):
                    self.hit("STATIC_SALT", node, nm)

    def visit_Compare(self, node):
        if any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops) and not self.has_compare_digest:
            sides = [node.left] + node.comparators
            srcs = [ast.unparse(s) for s in sides]
            if not any(isinstance(s, ast.Constant) for s in sides) and any(MAC_NAME.search(s) for s in srcs):
                self.hit("TIMING_COMPARE", node, " == ".join(srcs)[:80])
        self.generic_visit(node)


_REGEX_FALLBACK = [
    ("WEAK_HASH", r"hashlib\.(md5|sha1)\(|MD5\.new|SHA1\.new|hashes\.(MD5|SHA1)\("),
    ("ECB_MODE", r"MODE_ECB|modes\.ECB\("),
    ("WEAK_CIPHER", r"\b(DES3?|ARC4|ARC2|Blowfish|TripleDES)\b"),
    ("WEAK_RANDOM", r"\brandom\.(choice|choices|randint|random|randrange|getrandbits|sample)\("),
    ("TLS_NO_VERIFY", r"verify\s*=\s*False|CERT_NONE|check_hostname\s*=\s*False|_create_unverified_context"),
    ("OLD_TLS", r"PROTOCOL_(SSLv[23]|TLSv1(_1)?)\b"),
    ("JWT_NO_VERIFY", r"verify_signature['\"]\s*:\s*False|algorithms?\s*=\s*\[?['\"]none"),
    ("HARDCODED_KEY", r"(?i)\b(secret_key|api_key|aes_key|key|secret)\s*=\s*b?['\"][^'\"]{8,}['\"]"),
    ("STATIC_IV", r"(?i)\b(iv|nonce)\s*=\s*b?['\"]"),
]


def detect(code: str) -> list[dict]:
    """Return a list of findings: {label, cwe, severity, line, why, engine}."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        out = []
        for label, rx in _REGEX_FALLBACK:
            for m in re.finditer(rx, code):
                out.append((label, code[:m.start()].count("\n") + 1, "regex-fallback"))
        engine = "regex"
    else:
        v = _V(code)
        v.visit(tree)
        out, engine = v.hits, "ast"
    seen, res = set(), []
    for label, line, why in out:
        if (label, line) in seen:
            continue
        seen.add((label, line))
        cwe, sev, _ = LABELS[label]
        res.append({"label": label, "cwe": cwe, "severity": sev, "line": line, "why": why, "engine": engine})
    return res
