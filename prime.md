Directory structure:
└── primeintellect-ai-gpu-challenge/
    ├── README.md
    ├── common.py
    ├── coordinator.py
    ├── Dockerfile
    ├── Dockerfile_verifier
    ├── docs.md
    ├── prover.py
    └── verifier_service.py

================================================
FILE: README.md
================================================
# GPU Matrix Multiplication Verification Protocol

This protocol verifies a large matrix multiplication $C = A \times B$ performed by an untrusted GPU worker, while keeping the verifier’s computation much cheaper than $O(n^3)$.

## Core Idea

1. **Freivalds’ Algorithm**  
   - The verifier keeps matrices $A$ and $B$ (both $n \times n$).  
   - The worker computes $C = A \times B$.  
   - The verifier chooses a random challenge vector $\mathbf{r}$ (size $n$), which is kept secret until after $C$ is computed.  
   - The worker sends back $C \mathbf{r}$.  
   - The verifier computes $A \bigl(B \mathbf{r}\bigr)$ (only $O(n^2)$ complexity) and checks it against $C \mathbf{r}$.  
   - If they match (within numerical tolerance), $C$ is **very likely** correct.

2. **Merkle‐based Commitment**  
   - The worker **commits** to $C$ by building a Merkle tree over all rows of $C$ and sending the **Merkle root** as a binding commitment.  
   - Each row $C[i, :]$ is hashed to form a leaf of the Merkle tree.  
   - The tree is built level by level, and the final root acts as a single “fingerprint” of $C$.

3. **Spot Checks**  
   - After seeing the Merkle root, the verifier reveals $\mathbf{r}$.  
   - The worker returns $C \mathbf{r}$ along with **selected rows** of $C$ (for instance, randomly chosen), plus Merkle authentication paths that prove those rows are consistent with the committed root.  
   - The verifier recomputes those rows locally by doing $A[i,:] \times B$, an $O(n)$ operation per row, and checks them against the opened rows from $C$.  
   - The Merkle paths ensure the worker cannot produce inconsistent rows without invalidating the commitment root.

## Steps in Detail

1. **Verifier Picks (n, seed) Which Generate $A, B$:**  
   - Matrices $A, B \in \mathbb{R}^{n \times n}$ (or $\mathbb{F}_p$ in a finite field variant).

2. **Worker Computes $C$:**  
   - Receives (n, seed) and recreates $A, B$.  
   - Computes $C = A \times B$ (cost $O(n^3)$ GPU work).  
   - Builds the **Merkle tree** of row hashes $\{H(C[0,:]), \dots, H(C[n-1,:])\}$.  
   - Sends the **Merkle root** to the verifier.

3. **Verifier Sends Random Vector $\mathbf{r}$:**  
   - Kept secret until after the Merkle root is received.

4. **Worker Responds:**  
   - Sends $C \mathbf{r}$.  
   - Opens selected rows: for each chosen row $i$, sends row data and the Merkle authentication path.

5. **Verifier Verifies:**  
   - **Freivalds Check:** Compares $C \mathbf{r}$ to $A (B \mathbf{r})$ in $O(n^2)$ time.  
   - **Row Spot‐Check:** For each opened row $i$, verifies the Merkle path matches the root, then checks the row’s correctness by computing $A[i,:] \times B$.  

If all checks pass, the verifier concludes $C$ is correct with high probability, without doing a full $O(n^3)$ recomputation.



================================================
FILE: common.py
================================================
import time
import hashlib
import numpy as np
import torch
import numba
import struct

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
NTYPE = np.float32

MASK_64 = 0xFFFFFFFFFFFFFFFF
MASK_32 = 0xFFFFFFFF
INV_2_64 = float(2**64)
INV_2_32 = float(2**32)

R_TOL = 5e-4
A_TOL = 1e-4

# set numba threads to max half the number of cores to avoid oversubscription
numba.set_num_threads(numba.config.NUMBA_NUM_THREADS // 2)

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start} seconds")
        return result
    return wrapper

def safe_allclose(a, b, rtol=R_TOL, atol=A_TOL):
    if torch.isnan(a).any() or torch.isnan(b).any():
        return False
    if torch.isinf(a).any() or torch.isinf(b).any():
        return False
    return torch.allclose(a, b, rtol=rtol, atol=atol)

@numba.njit
def xorshift128plus_array(n, s0, s1):
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        x = s0
        y = s1
        s0 = y
        x ^= x << 23
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
        val = (s1 + y) & MASK_32
        val_float = float(val)
        out[i] = val_float * (1.0 / 4294967296.0)  # More precise than simple division
    return out, s0, s1

@numba.njit
def xorshift128plus_array64(n, s0, s1):
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = s0
        y = s1
        s0 = y
        x ^= x << 23
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
        val = (s1 + y) & MASK_64
        out[i] = val / INV_2_64
    return out, s0, s1

if DTYPE == torch.float32:
    xorshifter = xorshift128plus_array
else:
    xorshifter = xorshift128plus_array64

def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def create_rowhashes(n, master_seed):
    hashes = []
    cur = master_seed
    for _ in range(n):
        h = sha256_bytes(cur)
        hashes.append(h)
        cur = h
    return hashes, cur

def create_row_from_hash(n, seed_hash):
    fixed_seed1 = struct.unpack("<Q", seed_hash[:8])[0]
    fixed_seed2 = struct.unpack("<Q", seed_hash[8:16])[0]
    # keep only 63 bits because numba uses signed ints
    s0 = fixed_seed1 & (2**63 - 1)
    s1 = fixed_seed2 & (2**63 - 1)
    out, _, _ = xorshifter(n, s0, s1)
    return torch.from_numpy(out)

@numba.njit(parallel=True)
def _fill_matrix(n, seeds, out_mat):
    """
    Fills 'out_mat' in-place. 
    seeds is an (n, 2) array of (s0, s1) for each row.
    """
    for i in numba.prange(n):
        s0, s1 = seeds[i, 0], seeds[i, 1]
        row_vals, _, _ = xorshifter(n, s0, s1)
        out_mat[i, :] = row_vals

@timer
def create_deterministic_rowhash_matrix(n, master_seed):
    """
    1) Generate row hashes and convert them to seeds (s0, s1).
    2) Use a Numba-parallel for loop to fill a NumPy array row-by-row.
    3) Convert to a PyTorch Tensor on the desired device.
    """
    # Prepare row hashes
    row_hashes, next_hash = create_rowhashes(n, master_seed)

    # Convert each hash into two 63-bit seeds
    seeds_np = np.empty((n, 2), dtype=np.uint64)
    for i, h in enumerate(row_hashes):
        s0 = struct.unpack("<Q", h[:8])[0] & ((1 << 63) - 1)
        s1 = struct.unpack("<Q", h[8:16])[0] & ((1 << 63) - 1)
        seeds_np[i, 0] = s0
        seeds_np[i, 1] = s1

    # Allocate the result matrix in NumPy
    mat_np = np.empty((n, n), dtype=np.float32)

    # Parallel fill
    _fill_matrix(n, seeds_np, mat_np)

    # Convert to torch Tensor and move to device
    result_torch = torch.from_numpy(mat_np).to(DEVICE)
    return result_torch, next_hash



================================================
FILE: coordinator.py
================================================
import requests
import os
import json
import time
from eth_account import Account
from eth_account.messages import encode_defunct

VERIFIER_URL = os.getenv("VERIFIER_URL", "http://localhost:14141")
PROVER_URL   = os.getenv("PROVER_URL", "http://localhost:12121")

PRIVATE_KEY_HEX = os.getenv("PRIVATE_KEY_HEX")

# create timer decorator that prints the first string argument of the inner function as a string
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__}{args[0]} took {end - start} seconds")
        return result
    return wrapper

def sign_request(endpoint: str, data: dict | None, private_key_hex: str) -> str:
    """
    Matches the Rust logic:
      1) Sort JSON keys
      2) Convert to string
      3) message = endpoint + request_data_string
      4) EIP-191 sign (personal sign)
    Returns signature hex.
    """
    if data:
        # Sort the dict by keys
        sorted_keys = sorted(data.keys())
        sorted_data = {k: data[k] for k in sorted_keys}
        request_data_string = json.dumps(sorted_data)
    else:
        request_data_string = ""

    message_str = f"{endpoint}{request_data_string}"
    message = encode_defunct(text=message_str)
    signed = Account.sign_message(message, private_key=private_key_hex)
    return signed.signature.hex()

@timer
def post_signed_json(endpoint: str, data: dict | None = None) -> requests.Response:
    """
    1) Sort the data’s keys (like your Rust code does).
    2) Convert the sorted data to JSON.
    3) Create message_str = endpoint + sorted_json.
    4) EIP-191 sign that string (personal_sign).
    5) Send the result to the verifier with:
       - 'Content-Type: application/json'
       - 'X-Signature: <hex signature>'
       - 'X-Endpoint: <endpoint>'
    6) Return the requests.Response for further processing.
    """
    # 1) Sort the data
    data = data or {}
    sorted_keys = sorted(data.keys())
    sorted_dict = {k: data[k] for k in sorted_keys}

    # 2) Convert to JSON bytes
    sorted_json = json.dumps(sorted_dict)

    # 3) Concat endpoint + sorted_json
    message_str = f"{endpoint}{sorted_json}"

    # 4) EIP-191 sign
    message = encode_defunct(text=message_str)
    signed = Account.sign_message(message, private_key=PRIVATE_KEY_HEX)
    signature_hex = signed.signature.hex()

    # 5) POST to the verifier with the signature in headers
    url = VERIFIER_URL + endpoint
    headers = {
        "Content-Type": "application/json",
        "X-Signature": signature_hex,
    }

    return requests.post(url, json=sorted_dict, headers=headers)

@timer
def post_to_prover(endpoint: str, json: dict | None = None) -> requests.Response:
    return requests.post(f"{PROVER_URL}{endpoint}", json=json).json()

@timer
def get_from_prover(endpoint: str) -> requests.Response:
    return requests.get(f"{PROVER_URL}{endpoint}").json()

def run_protocol():
    # 1) Ask the verifier to init a new session: returns {session_id, n, master_seed}
    # init_resp = requests.post(f"{VERIFIER_URL}/init")
    params = {"n": 45000} # ~23GB of data
    params = {"n": 80000} # ~46GB of data
    params = {"n": 8192} # ~46GB of data
    init_resp = post_signed_json("/init", data=params)
    print(init_resp.text)
    init_data = init_resp.json()
    session_id = init_data["session_id"]
    n = init_data["n"]
    master_seed_hex = init_data["master_seed"]
    print("Session ID:", session_id, "   n =", n)

    # 2) Tell the prover to set A,B using n and master_seed
    setAB_data = post_to_prover(
        "/setAB",
        json={"n": n, "seed": master_seed_hex}
    )
    # check success
    if "status" not in setAB_data or setAB_data["status"] != "ok":
        print("Error setting A,B. Exiting.")
        return
    getCommitment_data = get_from_prover("/getCommitment")
    commitment_root_hex = getCommitment_data["commitment_root"]

    # 3) Pass the prover's commitment_root to the verifier -> get a challenge vector r
    challenge_resp = post_signed_json(
        "/commitment",
        data={
            "session_id": session_id,
            "commitment_root": commitment_root_hex
        }
    )
    challenge_data = challenge_resp.json()
    challenge_vector = challenge_data["challenge_vector"]

    # 4) Send challenge vector to the prover to compute C*r
    computeCR_data = post_to_prover(
        "/computeCR",
        json={"r": challenge_vector}
    )
    Cr = computeCR_data["Cr"]

    # 5) Ask the verifier which rows it wants to check, passing Cr for the Freivalds test
    #    Assume we encode Cr as a comma-separated string for the verifier’s rowchallenge endpoint
    # Cr_str = ",".join(str(x) for x in Cr)
    rowchallenge_resp = post_signed_json(
        "/row_challenge",
        data={
            "session_id": session_id,
            "Cr": Cr
        }
    )
    rowchallenge_data = rowchallenge_resp.json()
    freivalds_ok = rowchallenge_data["freivalds_ok"]
    if not freivalds_ok:
        print("Freivalds check failed. Exiting.")
        return

    # After receiving the spot_rows from /rowchallenge
    spot_rows = rowchallenge_data["spot_rows"]
    print("Freivalds check passed. Spot-checking rows:", spot_rows)

    # 6) Ask the prover for proofs of each row in one call: /getRowProofs
    #    Instead of calling /getRowProof for each row_idx, we can pass an array of row_idxs.
    rowproofs_data = post_to_prover(
        "/getRowProofs",
        json={"row_idxs": spot_rows}
    )

    # Suppose the prover's response is of the form:
    # { "rows": [
    #       { "row_idx": 5,  "row_data": [...], "merkle_path": [...] },
    #       { "row_idx": 11, "row_data": [...], "merkle_path": [...] },
    #       ...
    #   ]
    # }

    # 7) Send all these row proofs to the verifier in one go: /multi_row_check
    payload = {
        "session_id": session_id,
        "rows": rowproofs_data["rows"]
    }
    rowcheck_resp = post_signed_json("/multi_row_check", data=payload)
    rowcheck_result = rowcheck_resp.json()
    # Example response:
    # {
    #   "all_passed": true,
    #   "results": [
    #       { "row_idx": 5,  "pass": true },
    #       { "row_idx": 11, "pass": true }
    #   ]
    # }

    if not rowcheck_result["all_passed"]:
        print("One or more row checks failed:")
        for r in rowcheck_result["results"]:
            if not r["pass"]:
                print(f"Row check failed on row {r['row_idx']}")
        return

    print("All spot-checks succeeded. Verification complete.")

if __name__ == "__main__":
    run_protocol()



================================================
FILE: Dockerfile
================================================
# Use an official CUDA base image with Ubuntu
FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install Python + pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA (match CUDA version to the base image).
# The cu118 wheel is currently the latest for many PyTorch releases,
# but confirm it aligns with your container CUDA version or use a local wheel.
RUN pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install Flask for the REST API
RUN pip3 install --no-cache-dir tornado
RUN pip3 install --no-cache-dir joblib
RUN pip3 install --no-cache-dir numba
RUN pip3 install --no-cache-dir numpy
RUN pip3 install --no-cache-dir eth_account

# Create a working directory
WORKDIR /app

# Copy everything (including .py files) into /app
COPY ./prover.py /app
COPY ./common.py /app
COPY ./verifier_service.py /app

# Expose port 12121, 14141 for the servers
EXPOSE 12121
EXPOSE 14141

# Run the Prover application by default
CMD ["python3", "prover.py"]



================================================
FILE: Dockerfile_verifier
================================================
# Use an official CUDA base image with Ubuntu
FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install Python + pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA (match CUDA version to the base image).
# The cu118 wheel is currently the latest for many PyTorch releases,
# but confirm it aligns with your container CUDA version or use a local wheel.
RUN pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install Tornado for the REST API
RUN pip3 install --no-cache-dir tornado
RUN pip3 install --no-cache-dir eth_account
RUN pip3 install --no-cache-dir numba
RUN pip3 install --no-cache-dir numpy

# Create a working directory
WORKDIR /app

# Copy everything (including .py files) into /app
COPY ./verifier_service.py /app

# Expose port 14141 for the Tornado server
EXPOSE 14141

# Run the Prover application
CMD ["python3", "verifier_service.py"]



================================================
FILE: docs.md
================================================
# API Documentation

- [API Documentation](#api-documentation)
   * [Verifier](#verifier)
      + [Endpoints](#endpoints)
         - [1. `POST /init`](#1-post-init)
            * [Request Body](#request-body)
            * [Response Body](#response-body)
         - [2. `POST /commitment`](#2-post-commitment)
            * [Request Body](#request-body-1)
            * [Response Body](#response-body-1)
         - [3. `POST /row_challenge`](#3-post-row_challenge)
            * [Request Body](#request-body-2)
            * [Response Body](#response-body-2)
         - [4. `POST /multi_row_check`](#4-post-multi_row_check)
            * [Request Body](#request-body-3)
            * [Response Body](#response-body-3)
         - [5. `POST /clear`](#5-post-clear)
            * [Request Body](#request-body-4)
            * [Response Body](#response-body-4)
   * [Prover](#prover)
      + [Endpoints](#endpoints-1)
         - [1. `POST /setAB`](#1-post-setab)
            * [Request Body](#request-body-5)
            * [Response Body](#response-body-5)
         - [2. `GET /getCommitment`](#2-get-getcommitment)
            * [Response Body](#response-body-6)
         - [3. `POST /computeCR`](#3-post-computecr)
            * [Request Body](#request-body-6)
            * [Response Body](#response-body-7)
         - [4. `POST /getRowProofs`](#4-post-getrowproofs)
            * [Request Body](#request-body-7)
            * [Response Body](#response-body-8)

## Verifier

This Tornado-based HTTP API verifies matrix multiplication proofs using Freivalds' algorithm and spot-checking rows. Float vectors and matrices are transmitted in Base64 form (raw bytes of the underlying float array).

Requests must be signed with a private key that corresponds to the `ALLOWED_ADDRESS` environment variable.

### Endpoints

---

#### 1. `POST /init`

Creates a new session, generates a pair of deterministic matrices $\(A\)$ and $\(B\)$, and returns:
- A UUID-based `session_id`
- The square matrix dimension `n`
- A 16-byte `master_seed` (hex-encoded) for deterministic row generation.

##### Request Body
```json
{
  "n": 16384     // optional; default is 16384
}
```

##### Response Body
```json
{
  "session_id": "some-uuid",
  "n": 16384,
  "master_seed": "abcd1234..."   // hex-encoded
}
```

---

#### 2. `POST /commitment`

After instructing the prover to set $\(A\)$ and $\(B\)$, submit the Merkle root of the claimed product matrix $\(C\)$. The API returns a random challenge vector $\(r\)$ for Freivalds’ check.

##### Request Body
```json
{
  "session_id": "some-uuid",
  "commitment_root": "abcdef..."  // hex-encoded
}
```

##### Response Body
```json
{
  "challenge_vector": "base64-of-float-array"
}
```
The `challenge_vector` is a Base64-encoded float array, serialized in the same format used internally (e.g. 32-bit floats).

---

#### 3. `POST /row_challenge`

Sends the vector $\(C \cdot r\)$ so the API can perform Freivalds’ check. If the check passes, a set of row indices is returned for spot-checking.

##### Request Body
```json
{
  "session_id": "some-uuid",
  "Cr": "base64-of-float-array"
}
```
`Cr` is the Base64-encoded result of $\(C \cdot r\)$.

##### Response Body
```json
{
  "freivalds_ok": true,
  "spot_rows": [12, 999, ...]
}
```
If `freivalds_ok` is false, the spot rows list is empty.

---

#### 4. `POST /multi_row_check`

Performs a final spot-check on multiple rows. Each row’s content is Merkle-verified and numerically compared.

##### Request Body
```json
{
  "session_id": "some-uuid",
  "rows": [
    {
      "row_idx": 5,
      "row_data": "base64-of-float-array",
      "merkle_path": ["abcd...", "1234...", ...] // list of hex-encoded siblings
    },
    ...
  ]
}
```
- `row_data` is the Base64-encoded row of $\(C\)$ at `row_idx`.

##### Response Body
```json
{
  "all_passed": true,
  "results": [
    {
      "row_idx": 5,
      "pass": true
    },
    ...
  ]
}
```
If `all_passed` is false, at least one row failed verification. The session is freed after this call.

---

#### 5. `POST /clear`

Manually deletes a session prior to its completion or timeout, if the completion is no longer needed.

##### Request Body
```json
{
  "session_id": "some-uuid",
}
```

##### Response Body
```json
{
  "status": "ok"
}
```

If the session does not exist, this call will return a `500` error.

---

## Prover

This Tornado-based HTTP API manages a prover’s side of matrix multiplication verification. It deterministically constructs matrices $\(A\)$ and $\(B\)$ from a seed, computes $\(C = A \times B\)$, and constructs a Merkle tree over the rows of $\(C\)$. Floating-point vectors and rows are serialized in Base64 form to avoid numeric truncation.

### Endpoints

---

#### 1. `POST /setAB`

Generates matrices $\(A\)$ and $\(B\)$ of size $\(\text{n} \times \text{n}\)$ from a seed, computes $\(C = A \times B\)$, and constructs a Merkle tree over $\(C\)$.

##### Request Body
```json
{
  "n": 16384,
  "seed": "abcd1234..." // hex-encoded 16-byte seed
}
```

##### Response Body
```json
{
  "status": "ok"
}
```
On success, the product $\(C\)$ is held in memory, and the Merkle tree and its root are computed.

---

#### 2. `GET /getCommitment`

Returns the Merkle root of $\(C\)$. This root can be used for verifying the integrity of row proofs later.

##### Response Body
```json
{
  "commitment_root": "abcdef..."  // hex-encoded
}
```

---

#### 3. `POST /computeCR`

Computes $\(C \cdot r\)$. The challenge vector `r` is provided in Base64 form (raw bytes of the underlying float array).

##### Request Body
```json
{
  "r": "base64-of-float-array"
}
```

##### Response Body
```json
{
  "Cr": "base64-of-float-array"
}
```
The result is a Base64-encoded float array of length $\(n\)$.

---

#### 4. `POST /getRowProofs`

Given a list of row indices, returns the corresponding rows of $\(C\)$, along with their Merkle proof paths.

##### Request Body
```json
{
  "row_idxs": [12, 999, ...]
}
```

##### Response Body
```json
{
  "rows": [
    {
      "row_idx": 12,
      "row_data": "base64-of-float-array",        // The entire row's bytes
      "merkle_path": ["abcd...", "1234...", ...]  // hex-encoded list of siblings
    },
    ...
  ]
}
```
The `row_data` field is Base64-encoded raw bytes of the float array for row `row_idx`. The `merkle_path` is a list of hexadecimal-encoded sibling hashes proving that row’s membership under the `commitment_root`.


================================================
FILE: prover.py
================================================
import os
import base64

import torch
import numpy as np

import tornado.ioloop
import tornado.web
import tornado.escape

from common import (
    create_deterministic_rowhash_matrix,
    sha256_bytes,
    timer,
    DEVICE,
    DTYPE,
    NTYPE
)

PORT = int(os.getenv("PORT", 12121))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Global state
A = None
B = None
C = None
leaves = None
merkle_tree = None
commitment_root = None

# function that prints 0 the first time, then on subsequent calls, prints the time elapsed since the first call
def print_elapsed_time(msg=None, restart=False):
    if not DEBUG:
        return
    import time
    if restart:
        print_elapsed_time.start_time = time.time()
    if not hasattr(print_elapsed_time, 'start_time'):
        print_elapsed_time.start_time = time.time()
        print(msg, 0)
    else:
        print(msg, time.time() - print_elapsed_time.start_time)

def merkle_build_tree(leaves: list[bytes]) -> list[bytes]:
    level = leaves[:]
    tree = []
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i+1] if (i+1 < len(level)) else left
            combined = sha256_bytes(left + right)
            next_level.append(combined)
        tree.extend(level)
        level = next_level
    tree.extend(level)
    tree.reverse()
    return tree

def merkle_find_root(tree: list[bytes]) -> bytes:
    return tree[0] if tree else b''

@timer
def gen_merkle_data(C):
    leaves_temp = []
    for i in range(C.shape[0]):
        row_bytes = C[i,:].cpu().numpy().tobytes()
        leaves_temp.append(sha256_bytes(row_bytes))
    tree = merkle_build_tree(leaves_temp)
    root = merkle_find_root(tree)
    return root, tree, leaves_temp

def merkle_proof_path(idx: int, leaves_list: list[bytes], tree: list[bytes]) -> list[str]:
    path = []
    level = leaves_list[:]
    current_idx = idx
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i+1] if (i+1 < len(level)) else left
            combined = sha256_bytes(left + right)
            next_level.append(combined)

        sibling_idx = current_idx ^ 1
        if sibling_idx < len(level):
            path.append(level[sibling_idx].hex())

        current_idx //= 2
        level = next_level
    return path

@timer
def cuda_sync():
    if torch.cuda.is_available() and DEBUG:
        torch.cuda.synchronize()

def block_matmul(A, B, block_size=1024):
    """
    Block matrix multiplication that reduces FP32 rounding error accumulation
    """
    n = A.shape[0]
    result = torch.zeros((n, n), dtype=A.dtype, device=A.device)
    
    # Process matrix in blocks
    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)
            # Compute one block at a time
            block_result = torch.mm(A[i:i_end, :], B[:, j:j_end])
            result[i:i_end, j:j_end] = block_result
            
            # Optional CUDA sync after each major block to ensure precision
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    return result

def multi_gpu_block_matmul_from_gpu0(A_0, B_0, block_size=4096):
    """
    A_0 and B_0 both reside on 'cuda:0'.
    Splits A_0 by rows across all GPUs, replicates B_0, does block matmul locally,
    then gathers partial results on 'cuda:0' and concatenates.
    """
    n = A_0.shape[0]
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        # Fallback to single‐GPU block matmul on device 0
        return block_matmul(A_0, B_0, block_size)

    devices = [f'cuda:{i}' for i in range(num_gpus)]
    chunk_size = (n + num_gpus - 1) // num_gpus
    partial_results = []

    for i, dev in enumerate(devices):
        start = i * chunk_size
        end = min(start + chunk_size, n)
        if start >= end:
            break

        # "Shave down" A_0 by rows for this chunk
        A_chunk = A_0[start:end]
        A_chunk_dev = A_chunk.to(dev, non_blocking=True)
        B_dev = B_0.to(dev, non_blocking=True)

        # Local block matmul on GPU i
        partial_dev = block_matmul(A_chunk_dev, B_dev, block_size)

        # Copy partial result back to GPU 0
        partial_gpu0 = partial_dev.to('cuda:0', non_blocking=True)
        partial_results.append(partial_gpu0)

    # Concatenate the partial results (on GPU 0)
    return torch.cat(partial_results, dim=0)

@timer
def compute_C(A_0, B_0):
    """
    Automatically uses multi_gpu_block_matmul_from_gpu0 if multiple GPUs exist.
    A_0 and B_0 are assumed to be on 'cuda:0' already.
    """
    n = A_0.shape[0]
    if torch.cuda.device_count() > 1:
        return multi_gpu_block_matmul_from_gpu0(A_0, B_0, block_size=4096)
    else:
        if n > 4096:
            return block_matmul(A_0, B_0, 4096)
        else:
            return torch.mm(A_0, B_0)

class SetABHandler(tornado.web.RequestHandler):
    def post(self):
        global A, B, C, leaves, merkle_tree, commitment_root
        print_elapsed_time("SetABHandler", restart=True)
        data = tornado.escape.json_decode(self.request.body)
        n = data["n"]
        master_seed = bytes.fromhex(data["seed"])
        print_elapsed_time("SetABHandler: after decoding")

        A, next_seed = create_deterministic_rowhash_matrix(n, master_seed)
        B, _ = create_deterministic_rowhash_matrix(n, next_seed)

        cuda_sync()
        C = compute_C(A, B)

        print_elapsed_time("SetABHandler: after computing C")

        commitment_root, merkle_tree, leaves = gen_merkle_data(C)

        print_elapsed_time("SetABHandler: after generating merkle data")
        self.write({"status": "ok"})

class GetCommitmentHandler(tornado.web.RequestHandler):
    def get(self):
        global commitment_root
        self.write({"commitment_root": commitment_root.hex()})

class ComputeCRHandler(tornado.web.RequestHandler):
    def post(self):
        global C
        data = tornado.escape.json_decode(self.request.body)

        # Encoding of raw buffer via base64 to reduce truncation errors
        r_b64 = data["r"]
        r_bytes = base64.b64decode(r_b64)
        r_array = np.frombuffer(r_bytes, dtype=NTYPE)
        r_t = torch.from_numpy(r_array.copy()).to(DEVICE)

        # r_t = torch.tensor(r_list, dtype=DTYPE, device=DEVICE)
        C_t = C # C.to(DEVICE)
        Cr_t = torch.matmul(C_t, r_t)
        # Cr = Cr_t.cpu().tolist()
        
        # Encode Cr to base64
        Cr_bytes = Cr_t.cpu().numpy().tobytes()
        Cr_b64 = base64.b64encode(Cr_bytes).decode()

        self.write({"Cr": Cr_b64})

class GetRowProofsHandler(tornado.web.RequestHandler):
    def post(self):
        global C, leaves, merkle_tree
        data = tornado.escape.json_decode(self.request.body)
        row_idxs = data["row_idxs"]
        rows_output = []
        for row_idx in row_idxs:
            # encode row_data to base64
            row_data = C[row_idx, :].cpu().numpy().tobytes()
            row_data_b64 = base64.b64encode(row_data).decode()
            path = merkle_proof_path(row_idx, leaves, merkle_tree)
            rows_output.append({
                "row_idx": row_idx,
                "row_data": row_data_b64,
                "merkle_path": path
            })
        self.write({"rows": rows_output})

def make_app():
    return tornado.web.Application([
        (r"/setAB", SetABHandler),
        (r"/getCommitment", GetCommitmentHandler),
        (r"/computeCR", ComputeCRHandler),
        (r"/getRowProofs", GetRowProofsHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(PORT)
    print(f"Prover API listening on port {PORT}, device: {DEVICE}, dtype: {DTYPE}")
    tornado.ioloop.IOLoop.current().start()



================================================
FILE: verifier_service.py
================================================
import os
import secrets
import uuid
import json
import base64
import time

import torch
import numpy as np

import tornado.ioloop
import tornado.web
import tornado.escape

from eth_account import Account
from eth_account.messages import encode_defunct

from common import (
    create_deterministic_rowhash_matrix,
    create_row_from_hash,
    sha256_bytes,
    safe_allclose,
    DTYPE,
    NTYPE,
    A_TOL,
    R_TOL,
    DEVICE
)
AUTHORIZED_ADDRESS = os.getenv("AUTHORIZED_ADDRESS", "").lower()
# e.g. "0xAbCd1234..." the address derived from your private key

VERIFIER_PORT = int(os.getenv("VERIFIER_PORT", 14141))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

GB = 1024**3

# In-memory sessions keyed by session_id
SESSIONS = {}
CURRENT_MEMORY = 0
MAX_MEMORY = int(os.getenv("MAX_MEMORY", 100*GB))  # 100 GB
SESSION_TIMEOUT = 300  # 5 minutes

# -------------------------------------------------------
# Merkle proof helpers
# -------------------------------------------------------

def merkle_verify_leaf(leaf: bytes, idx: int, path: list[str], root: bytes) -> bool:
    current = leaf
    current_idx = idx
    for sibling_hex in path:
        sibling = bytes.fromhex(sibling_hex)
        if (current_idx % 2) == 0:  # even => left child
            current = sha256_bytes(current + sibling)
        else:  # odd => right child
            current = sha256_bytes(sibling + current)
        current_idx //= 2
    return current == root

def check_freivals(A, B, Cr, r):
    """
    Freivalds check: verify A(B*r) == C*r without computing C= A*B fully.
    """
    # Use accumulation pattern that reduces error
    x = B.matmul(r)
    
    # Compare with residual-based approach
    check = A.matmul(x)
    
    # Kahan summation pattern for reduced error when checking difference
    # residual = check - Cr
    # residual_norm = torch.norm(residual)
    # sum_norm = torch.norm(check) + 1e-10  # Avoid division by zero
    
    # Check both relative and absolute error
    # relative_error = residual_norm / sum_norm
    # absolute_error = residual_norm
    
    # For debugging: print the actual errors
    # print(f"Relative error: {relative_error.item()}, Absolute error: {absolute_error.item()}")

    return safe_allclose(check, Cr, rtol=R_TOL, atol=A_TOL)

def check_row_correctness(A_row, B, claimed_row):
    """
    More numerically stable row verification for float32
    """
    # Compute product in blocks to reduce accumulation errors
    n = B.shape[1]
    local_check_row = torch.zeros(n, dtype=A_row.dtype, device=A_row.device)
    
    # Process in smaller blocks to reduce error accumulation
    block_size = min(1024, n)
    for j in range(0, n, block_size):
        j_end = min(j + block_size, n)
        local_check_row[j:j_end] = torch.matmul(A_row, B[:, j:j_end])
    
    # Check if rows match with appropriate tolerance
    return torch.allclose(local_check_row, claimed_row, rtol=R_TOL, atol=A_TOL)

# -------------------------------------------------------
# Tornado handlers
# -------------------------------------------------------

class BaseHandler(tornado.web.RequestHandler):
    def prepare(self):
        self._kill_timeout = tornado.ioloop.IOLoop.current().call_later(
            SESSION_TIMEOUT, self._kill_connection
        )
        # Parse JSON once, store in self.body_dict for child handlers
        if self.request.body:
            try:
                self.body_dict = tornado.escape.json_decode(self.request.body)
            except json.JSONDecodeError:
                self.set_status(400)
                self.write({"error": "Invalid JSON"})
                self.finish()
                return
        else:
            self.body_dict = {}

        # Extract signature from headers
        signature_hex = self.request.headers.get("X-Signature", "")
        if not signature_hex:
            self.set_status(401)
            self.write({"error": "Missing X-Signature header"})
            self.finish()
            return

        # Rebuild the same message the coordinator used: self.request.path + sorted JSON
        endpoint = self.request.path
        sorted_keys = sorted(self.body_dict.keys())
        sorted_data = {k: self.body_dict[k] for k in sorted_keys}
        request_data_string = json.dumps(sorted_data)

        message_str = f"{endpoint}{request_data_string}"
        message = encode_defunct(text=message_str)
        
        # disable signature verification in DEBUG mode
        if not DEBUG:
            # Recover address from signature
            try:
                recovered_address = Account.recover_message(
                    message, 
                    signature=bytes.fromhex(signature_hex)
                )
            except:
                self.set_status(401)
                self.write({"error": "Signature recovery failed"})
                self.finish()
                return
            
            # Check against our authorized address
            if recovered_address.lower() != AUTHORIZED_ADDRESS:
                self.set_status(401)
                self.write({"error": "Unauthorized signer"})
                self.finish()
                return

        # If all good, proceed. Child handlers can access self.body_dict.
    
    def on_finish(self):
        if hasattr(self, "_kill_timeout"):
            tornado.ioloop.IOLoop.current().remove_timeout(self._kill_timeout)

    def _kill_connection(self):
        if not self._finished:
            self.set_status(408)
            self.finish("Request exceeded maximum duration.")

class InitHandler(BaseHandler):
    """
    1) Create a new session, generate random A,B and return session_id plus (n, master_seed).
    The coordinator will pass (n, master_seed) to the prover's /setAB.
    """
    def post(self):
        global CURRENT_MEMORY

        body = self.body_dict
        n = body.get("n", 16384)  # default 16384 if not provided
        if n > 2**18:
            n = 2**18

        # calculate cost of new session
        # n^2 * size_of_type * 2.5_matrices (A, B) + some overhead
        memory_cost = n**2 * np.dtype(NTYPE).itemsize * 3
        if CURRENT_MEMORY + memory_cost > MAX_MEMORY:
            # try to prune stale sessions
            sessions_to_delete = []
            for session_id, session_data in SESSIONS.items():
                if time.time() - session_data["start_time"] > SESSION_TIMEOUT:
                    sessions_to_delete.append(session_id)
                    CURRENT_MEMORY -= session_data["memory_cost"]
            for session_id in sessions_to_delete:
                del SESSIONS[session_id]

        # if we've still not up enough memory, error
        if CURRENT_MEMORY + memory_cost > MAX_MEMORY:
            self.write({"error": f"Memory limit exceeded, wait up to {SESSION_TIMEOUT} seconds for a session to expire"})
            return

        master_seed = secrets.token_bytes(16)
        A, next_seed = create_deterministic_rowhash_matrix(n, master_seed)
        B, _         = create_deterministic_rowhash_matrix(n, next_seed)

        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {
            "n": n,
            "master_seed": master_seed,
            "A": A,
            "B": B,
            "commitment_root": None,
            "r": None,
            "Cr": None,
            "spot_rows": None,
            "memory_cost": memory_cost,
            "start_time": time.time()
        }

        # Return session info for the coordinator
        self.write({
            "session_id": session_id,
            "n": n,
            "master_seed": master_seed.hex()
        })

class CommitmentHandler(BaseHandler):
    """
    2) After the coordinator has told the prover to set A,B,
       the coordinator calls POST /commitment with { session_id, commitment_root }.
       The verifier stores commitment_root and returns the random challenge vector r.
    """
    def post(self):
        body = self.body_dict
        session_id = body.get("session_id")
        commitment_root_hex = body.get("commitment_root")

        if not session_id or session_id not in SESSIONS:
            self.write({"error": "Invalid or missing session_id"})
            return
        if not commitment_root_hex:
            self.write({"error": "Missing commitment_root"})
            return

        session_data = SESSIONS[session_id]
        session_data["commitment_root"] = bytes.fromhex(commitment_root_hex)

        # Generate random challenge vector r
        n = session_data["n"]
        challenge_seed = secrets.token_bytes(16)
        r = create_row_from_hash(n, challenge_seed)
        session_data["r"] = r

        # Encode r as a base64 string to reduce truncation errors
        r_bytes = r.cpu().numpy().tobytes()
        r_b64 = base64.b64encode(r_bytes).decode()

        # Return the challenge vector as a list
        self.write({"challenge_vector": r_b64})

class RowChallengeHandler(BaseHandler):
    """
    3) The coordinator calls /row_challenge with { session_id, Cr } after the prover computed C*r.
       The verifier does the Freivalds check with stored A,B,r. If passes, picks row(s) for spot-check.
    """
    def post(self):
        body = self.body_dict
        session_id = body.get("session_id")
        if not session_id or session_id not in SESSIONS:
            self.write({"error": "Invalid or missing session_id"})
            return
        
        # Encoding of raw buffer via base64 to reduce truncation errors
        Cr_b64 = body.get("Cr")
        Cr_bytes = base64.b64decode(Cr_b64)
        Cr_array = np.frombuffer(Cr_bytes, dtype=NTYPE)
        Cr_tensor = torch.from_numpy(Cr_array.copy())
        
        # Cr_string_list = self.body_dict.get("Cr", [])
        # Cr_list = [float(s) for s in Cr_string_list]
        # Cr_tensor = torch.tensor(Cr_list, dtype=DTYPE)

        # Cr_list = body.get("Cr")
        # if Cr_list is None:
        #    self.write({"error": "Missing Cr"})
        #    return

        session_data = SESSIONS[session_id]
        A, B, r = session_data["A"], session_data["B"], session_data["r"]
        if (A is None) or (B is None) or (r is None):
            self.write({"error": "Session missing A,B,r"})
            return


        # Perform Freivalds
        freivalds_ok = check_freivals(A, B, Cr_tensor.to(DEVICE), r.to(DEVICE))
        if not freivalds_ok:
            self.write({"freivalds_ok": False, "spot_rows": []})
            return

        # If ok, pick some rows to spot check
        k = 10
        n = session_data["n"]
        chosen_rows = []
        while len(set(chosen_rows)) < k:
            chosen_rows = [secrets.randbelow(n) for _ in range(k)]

        session_data["spot_rows"] = chosen_rows
        self.write({
            "freivalds_ok": True,
            "spot_rows": chosen_rows
        })

class MultiRowCheckHandler(BaseHandler):
    """
    5) Coordinator calls /multi_row_check with JSON:
       {
         "session_id": "...",
         "rows": [
           {
             "row_idx": 5,
             "row_data": [...],
             "merkle_path": [...]
           },
           ...
         ]
       }
       Checks each row. Returns {
         "all_passed": bool,
         "results": [ {"row_idx": ..., "pass": bool}, ... ]
       }
    """
    def post(self):
        body = self.body_dict
        session_id = body.get("session_id")
        rows_info = body.get("rows", [])

        if not session_id or session_id not in SESSIONS:
            self.write({
                "all_passed": False,
                "error": "Invalid or missing session_id",
                "results": []
            })
            return

        session_data = SESSIONS[session_id]
        A, B, root = session_data["A"], session_data["B"], session_data["commitment_root"]
        if A is None or B is None or root is None:
            self.write({
                "all_passed": False,
                "error": "Missing required session data (A,B,root)",
                "results": []
            })
            return

        results = []
        all_passed = True

        for row_obj in rows_info:
            row_idx = row_obj.get("row_idx")
            row_data_b64 = row_obj.get("row_data")
            row_data_bytes = base64.b64decode(row_data_b64)
            row_data = np.frombuffer(row_data_bytes, dtype=NTYPE)
            merkle_path = row_obj.get("merkle_path")

            # Basic checks
            if row_idx is None or row_data is None or merkle_path is None:
                results.append({"row_idx": row_idx, "pass": False})
                all_passed = False
                continue

            # Convert to tensor
            row_data_tensor = torch.tensor(row_data, dtype=DTYPE)

            # 1) Merkle verify
            leaf_bytes = sha256_bytes(row_data_tensor.cpu().numpy().tobytes())
            path_ok = merkle_verify_leaf(leaf_bytes, row_idx, merkle_path, root)
            if not path_ok:
                results.append({"row_idx": row_idx, "pass": False})
                all_passed = False
                continue

            # 2) Row correctness
            row_of_A = A[row_idx, :]
            # local_check_row = row_of_A.matmul(B)
            # row_checks_ok = torch.allclose(local_check_row, row_data_tensor, rtol=R_TOL, atol=A_TOL)
            row_checks_ok = check_row_correctness(row_of_A.to(DEVICE), B, row_data_tensor.to(DEVICE))

            passed = bool(path_ok and row_checks_ok)
            if not passed:
                all_passed = False

            results.append({"row_idx": row_idx, "pass": passed})

        # delete session to free up memory
        del SESSIONS[session_id]

        self.write({
            "all_passed": all_passed,
            "results": results
        })

class ClearHandler(BaseHandler):
    """
    Clear a session by session_id"
    """
    def post(self):
        body = self.body_dict
        session_id = body.get("session_id")
        if not session_id or session_id not in SESSIONS:
            self.write({"error": "Invalid or missing session_id"})
            return

        session_data = SESSIONS[session_id]
        memory_cost = session_data["memory_cost"]
        del SESSIONS[session_id]
        global CURRENT_MEMORY
        CURRENT_MEMORY -= memory_cost

        self.write({"status": "ok"})


def make_app():
    return tornado.web.Application([
        (r"/init",            InitHandler),
        (r"/commitment",      CommitmentHandler),
        (r"/row_challenge",   RowChallengeHandler),
        (r"/multi_row_check", MultiRowCheckHandler),
        (r"/clear",           ClearHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(VERIFIER_PORT)
    print(f"Verifier API running on port {VERIFIER_PORT}, dtype: {DTYPE}")
    tornado.ioloop.IOLoop.current().start()


