"""
Verification script for AFINE SDK implementation.
"""

import sys
import os

sys.path.insert(0, "python")

print("=" * 60)
print("BASILICA AFINE SDK VERIFICATION")
print("=" * 60)

errors = []

print("\n1. Testing module imports...")
try:
    from basilica.afine import Service, serve, Client, create
    from basilica.afine import BasilicaAPIClient, DockerManager, StatePersistence
    from basilica.afine import RPCRequest, RPCResponse, ErrorResponse
    print("   ✓ All imports successful")
except Exception as e:
    errors.append(f"Import error: {e}")
    print(f"   ✗ Import failed: {e}")

print("\n2. Testing Service base class...")
try:
    from basilica.afine import Service

    class TestEnv(Service):
        def reset(self, seed=None):
            return "test"

        def step(self, action):
            return "test", 1.0, False, False, {}

    env = TestEnv()
    assert env.reset() == "test"
    print("   ✓ Service base class works")
except Exception as e:
    errors.append(f"Service error: {e}")
    print(f"   ✗ Service failed: {e}")

print("\n3. Testing Pydantic models...")
try:
    from basilica.afine.models import RPCRequest, RPCResponse, ErrorResponse

    req = RPCRequest(args=[1, 2], kwargs={"test": "value"})
    assert req.args == [1, 2]
    assert req.kwargs == {"test": "value"}

    resp = RPCResponse(result="success")
    assert resp.result == "success"

    err = ErrorResponse(error="TestError", detail="Test detail")
    assert err.error == "TestError"
    print("   ✓ Pydantic models work")
except Exception as e:
    errors.append(f"Pydantic error: {e}")
    print(f"   ✗ Pydantic models failed: {e}")

print("\n4. Testing StatePersistence...")
try:
    import tempfile
    from pathlib import Path
    from basilica.afine import StatePersistence

    with tempfile.TemporaryDirectory() as tmpdir:
        persistence = StatePersistence(state_dir=Path(tmpdir))

        class TestObj:
            def __init__(self):
                self.value = 42

        obj = TestObj()
        persistence.save_state(obj, name="test")

        loaded = persistence.load_state(name="test")
        assert loaded["value"] == 42
    print("   ✓ StatePersistence works")
except Exception as e:
    errors.append(f"StatePersistence error: {e}")
    print(f"   ✗ StatePersistence failed: {e}")

print("\n5. Testing Client proxy structure...")
try:
    from basilica.afine import Client

    client = Client(
        rental_id="test",
        base_url="http://localhost:8000",
        rental_secret="test-secret"
    )

    assert client._rental_id == "test"
    assert client._rental_secret == "test-secret"
    assert not client._closed

    client.close()
    assert client._closed
    print("   ✓ Client proxy works")
except Exception as e:
    errors.append(f"Client error: {e}")
    print(f"   ✗ Client failed: {e}")

print("\n6. Checking example files...")
example_files = [
    "examples/afine/mathenv/service.py",
    "examples/afine/satenv/service.py",
    "examples/afine/cartpole/service.py",
]

for example in example_files:
    if os.path.exists(example):
        print(f"   ✓ {example}")
    else:
        errors.append(f"Missing example: {example}")
        print(f"   ✗ {example} missing")

print("\n7. Checking documentation...")
doc_files = [
    "python/basilica/afine/README.md",
]

for doc in doc_files:
    if os.path.exists(doc):
        print(f"   ✓ {doc}")
    else:
        errors.append(f"Missing doc: {doc}")
        print(f"   ✗ {doc} missing")

print("\n" + "=" * 60)
if errors:
    print(f"VERIFICATION FAILED ({len(errors)} errors)")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)
else:
    print("VERIFICATION SUCCESSFUL")
    print("All components implemented and working!")
    sys.exit(0)
