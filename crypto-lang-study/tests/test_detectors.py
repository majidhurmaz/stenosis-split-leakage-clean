"""Known-answer tests for the custom detectors. Run: python -m pytest tests  (or python tests/test_detectors.py)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from detectors import detect  # noqa: E402

CASES = [
    # (code, labels that MUST be found, labels that must NOT be found)
    ("import hashlib\nh = hashlib.md5(data).hexdigest()", {"WEAK_HASH"}, set()),
    ("import hashlib\nh = hashlib.md5(data, usedforsecurity=False)", set(), {"WEAK_HASH"}),
    ("import hashlib\ndef s(password):\n    return hashlib.sha256(password.encode()).hexdigest()", {"FAST_PASSWORD_HASH"}, set()),
    ("import hashlib, os\ndef s(password):\n    salt=os.urandom(16)\n    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)",
     {"LOW_KDF_ITERATIONS"}, {"FAST_PASSWORD_HASH", "STATIC_SALT"}),
    ("import hashlib, os\nk = hashlib.pbkdf2_hmac('sha256', pw, os.urandom(16), 600000)", set(), {"LOW_KDF_ITERATIONS"}),
    ("from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC\nk=PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=b'fixedsalt', iterations=1000)",
     {"LOW_KDF_ITERATIONS", "STATIC_SALT"}, set()),
    ("import bcrypt\nh = bcrypt.hashpw(password.encode(), bcrypt.gensalt())\nok = bcrypt.checkpw(p, h)", set(), {"FAST_PASSWORD_HASH", "TIMING_COMPARE"}),
    ("from Crypto.Cipher import AES\nc = AES.new(key, AES.MODE_ECB)", {"ECB_MODE"}, set()),
    ("from Crypto.Cipher import DES\nc = DES.new(key, DES.MODE_CBC, iv)", {"WEAK_CIPHER"}, set()),
    ("from Crypto.Cipher import AES\niv = b'1234567890123456'\nc = AES.new(key, AES.MODE_CBC, iv)", {"STATIC_IV", "UNAUTH_ENCRYPTION"}, set()),
    ("from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes\nc = Cipher(algorithms.AES(k), modes.CBC(b'\\x00'*16))", {"STATIC_IV"}, set()),
    ("from cryptography.hazmat.primitives.ciphers.aead import AESGCM\nimport os\nn=os.urandom(12)\nct=AESGCM(key).encrypt(n, d, None)", set(), {"STATIC_IV", "HARDCODED_KEY"}),
    ("SECRET_KEY = 'super-secret-value-123'\n", {"HARDCODED_KEY"}, set()),
    ("app.config['SECRET_KEY'] = 'abcdefgh12345'\n", {"HARDCODED_KEY"}, set()),
    ("import os\nSECRET_KEY = os.environ['SECRET_KEY']\n", set(), {"HARDCODED_KEY"}),
    ("import random, string\nt=''.join(random.choice(string.ascii_letters) for _ in range(32))", {"WEAK_RANDOM"}, set()),
    ("import secrets\nt=secrets.token_urlsafe(32)", set(), {"WEAK_RANDOM"}),
    ("from random import randint\notp = randint(100000, 999999)", {"WEAK_RANDOM"}, set()),
    ("from cryptography.hazmat.primitives.asymmetric import rsa\nk=rsa.generate_private_key(public_exponent=65537, key_size=1024)", {"SHORT_RSA_KEY"}, set()),
    ("from Crypto.PublicKey import RSA\nk = RSA.generate(2048)", set(), {"SHORT_RSA_KEY"}),
    ("import requests\nr = requests.post(url, json=d, verify=False)", {"TLS_NO_VERIFY"}, set()),
    ("import requests\nr = requests.post(url, json=d, verify='ca.pem')", set(), {"TLS_NO_VERIFY"}),
    ("import ssl\nctx = ssl.create_default_context()\nctx.check_hostname = False\nctx.verify_mode = ssl.CERT_NONE", {"TLS_NO_VERIFY"}, set()),
    ("import ssl\nctx = ssl.SSLContext(ssl.PROTOCOL_TLSv1)", {"OLD_TLS"}, set()),
    ("import jwt\np = jwt.decode(tok, options={'verify_signature': False})", {"JWT_NO_VERIFY"}, set()),
    ("import jwt\np = jwt.decode(tok, 'mysecretkey123', algorithms=['HS256'])", {"HARDCODED_KEY"}, {"JWT_NO_VERIFY"}),
    ("import hmac, hashlib\nexpected = hmac.new(k, body, hashlib.sha256).hexdigest()\nif expected == signature:\n    ok()", {"TIMING_COMPARE"}, set()),
    ("import hmac, hashlib\nexpected = hmac.new(k, body, hashlib.sha256).hexdigest()\nif hmac.compare_digest(expected, signature):\n    ok()", set(), {"TIMING_COMPARE"}),
    ("from cryptography.hazmat.primitives import serialization\npem=k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())", {"PRIVKEY_UNENCRYPTED"}, set()),
    ("def broken(:\n  iv = b'aaaaaaaaaaaaaaaa'\n  AES.MODE_ECB", {"STATIC_IV", "ECB_MODE"}, set()),  # syntax error -> regex fallback
]


def test_all():
    failures = []
    for code, must, must_not in CASES:
        got = {h["label"] for h in detect(code)}
        if not must <= got or got & must_not:
            failures.append((code.splitlines()[-1][:70], sorted(must - got), sorted(got & must_not), sorted(got)))
    for f in failures:
        print("FAIL:", f)
    assert not failures, f"{len(failures)} failing cases"


if __name__ == "__main__":
    test_all()
    print(f"all {len(CASES)} detector cases pass")
