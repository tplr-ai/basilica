// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {CollateralUpgradeable} from "../src/CollateralUpgradeable.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";

// ---------------------------------------------------------------------------
// NodeOwnershipEdgeCasesTest
// Focused tests for node ownership (nodeToMiner) and swap-and-pop (activeNodeKeys)
// edge cases. Each function covers one "when" branch from the .tree spec.
// ---------------------------------------------------------------------------

/// NodeOwnershipEdgeCasesTest
contract NodeOwnershipEdgeCasesTest is Test {
    CollateralUpgradeable public collateral;
    CollateralUpgradeable public implementation;
    ERC1967Proxy public proxy;

    // Precompile addresses (Bittensor EVM)
    address constant ADDRESS_MAPPING_PRECOMPILE = 0x000000000000000000000000000000000000080C;
    address constant STAKING_V2_PRECOMPILE = 0x0000000000000000000000000000000000000805;

    // Contract config
    uint16 constant NETUID = 42;
    address constant TRUSTEE = address(0x123);
    address constant ADMIN = address(0x456);
    uint256 constant MIN_DEPOSIT = 1 ether;
    uint64 constant DECISION_TIMEOUT = 3600; // 1 hour
    bytes32 constant ALPHA_HOTKEY = bytes32(uint256(1));

    // Test accounts
    address constant ALICE = address(0x789);
    address constant BOB = address(0xABC);
    address constant CHARLIE = address(0xDEF);
    address constant DAVE = address(0x1111);

    // Node identifiers
    bytes32 constant HOTKEY_1 = bytes32(uint256(100));
    bytes32 constant HOTKEY_2 = bytes32(uint256(101));
    bytes32 constant HOTKEY_3 = bytes32(uint256(102));
    bytes32 constant HOTKEY_4 = bytes32(uint256(103));
    bytes16 constant EXECUTOR_ID_1 = bytes16(uint128(1));
    bytes16 constant EXECUTOR_ID_2 = bytes16(uint128(2));
    bytes16 constant EXECUTOR_ID_3 = bytes16(uint128(3));
    bytes16 constant EXECUTOR_ID_4 = bytes16(uint128(4));

    // Evidence params (arbitrary, required by API)
    string constant TEST_URL = "https://example.com/proof";
    bytes32 constant TEST_SHA256 = bytes32(0x1234567890123456789012345678901212345678901234567890123456789012);

    // ---------------------------------------------------------------------------
    // Setup
    // ---------------------------------------------------------------------------

    function setUp() public {
        // Deploy precompile mocks
        _AddressMappingMock addrMock = new _AddressMappingMock();
        vm.etch(ADDRESS_MAPPING_PRECOMPILE, address(addrMock).code);
        _StakingV2Mock stakingMock = new _StakingV2Mock();
        vm.etch(STAKING_V2_PRECOMPILE, address(stakingMock).code);

        // Deploy implementation + proxy
        implementation = new CollateralUpgradeable();
        bytes memory initData = abi.encodeWithSelector(
            CollateralUpgradeable.initialize.selector,
            NETUID,
            TRUSTEE,
            MIN_DEPOSIT,
            MIN_DEPOSIT,
            DECISION_TIMEOUT,
            ADMIN,
            ALPHA_HOTKEY,
            true,
            true
        );
        proxy = new ERC1967Proxy(address(implementation), initData);
        collateral = CollateralUpgradeable(payable(address(proxy)));

        // Pre-seed HOTKEY_1 ownership to ALICE so deposits don't require EOA check
        _seedNodeOwner(HOTKEY_1, EXECUTOR_ID_1, ALICE);

        // Fund test accounts
        vm.deal(ALICE, 100 ether);
        vm.deal(BOB, 100 ether);
        vm.deal(CHARLIE, 100 ether);
        vm.deal(DAVE, 100 ether);
    }

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    /// @dev Storage slot for nodeToMiner[hotkey][nodeId] (slot 5 in contract layout)
    function _nodeToMinerSlot(bytes32 hotkey, bytes16 nodeId) internal pure returns (bytes32) {
        uint256 nodeToMinerSlot = 5;
        bytes32 levelOne = keccak256(abi.encode(hotkey, nodeToMinerSlot));
        return keccak256(abi.encode(nodeId, levelOne));
    }

    /// @dev Directly set nodeToMiner without going through deposit (bypasses EOA check)
    function _seedNodeOwner(bytes32 hotkey, bytes16 nodeId, address owner) internal {
        vm.store(address(collateral), _nodeToMinerSlot(hotkey, nodeId), bytes32(uint256(uint160(owner))));
    }

    function _allCollaterals() internal view returns (CollateralUpgradeable.NodeCollateral[] memory) {
        return collateral.getAllCollaterals(0, type(uint256).max);
    }

    function _allReclaims() internal view returns (CollateralUpgradeable.ReclaimInfo[] memory) {
        return collateral.getAllReclaims(0, type(uint256).max);
    }

    // ---------------------------------------------------------------------------
    // Test: when previous miner has finalized reclaim
    //
    // it clears nodeToMiner after finalize
    // it removes node from active tracking after finalize
    // it allows a new miner to claim the same node
    // ---------------------------------------------------------------------------

    function test_WhenPreviousMinerHasFinalizedReclaim() external {
        // Alice deposits 5 ether (HOTKEY_1 already seeded to ALICE in setUp)
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Alice initiates reclaim
        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        // Warp past decision timeout and finalize
        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);
        collateral.finalizeReclaim(0);

        // it clears nodeToMiner after finalize
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), address(0));

        // it removes node from active tracking after finalize
        assertEq(collateral.getActiveNodeCount(), 0);
        assertEq(_allCollaterals().length, 0);

        // it allows a new miner to claim the same node
        // Bob is a new EOA claimant; both tx.origin and msg.sender must be BOB
        vm.prank(BOB, BOB);
        collateral.deposit{value: 3 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), BOB);
        assertEq(collateral.getActiveNodeCount(), 1);
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 3 ether);
    }

    // ---------------------------------------------------------------------------
    // Test: when another miner tries to deposit during active reclaim
    //
    // it preserves ownership for original miner
    // it reverts with NodeNotOwned
    // ---------------------------------------------------------------------------

    function test_WhenAnotherMinerTriesToDepositDuringActiveReclaim() external {
        // Alice deposits and starts a reclaim (NOT finalized)
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        // it preserves ownership for original miner during pending reclaim
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), ALICE);

        // it reverts with NodeNotOwned when another miner tries to deposit
        vm.prank(BOB);
        vm.expectRevert(abi.encodeWithSelector(CollateralUpgradeable.NodeNotOwned.selector));
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);
    }

    // ---------------------------------------------------------------------------
    // Test: when trustee fully slashes with no pending reclaims
    //
    // it clears nodeToMiner after full slash
    // it removes node from active tracking after full slash
    // it allows a new miner to claim the same node after full slash
    // ---------------------------------------------------------------------------

    function test_WhenTrusteeFullySlashesWithNoPendingReclaims() external {
        // Alice deposits 5 ether
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Trustee slashes the full amount (no pending reclaims)
        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 5 ether, 0, TEST_URL, TEST_SHA256);

        // it clears nodeToMiner after full slash
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), address(0));

        // it removes node from active tracking after full slash
        assertEq(collateral.getActiveNodeCount(), 0);
        assertEq(_allCollaterals().length, 0);

        // it allows a new miner to claim the same node after full slash
        vm.prank(BOB, BOB);
        collateral.deposit{value: 3 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), BOB);
        assertEq(collateral.getActiveNodeCount(), 1);
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 3 ether);
    }

    // ---------------------------------------------------------------------------
    // Test: when trustee partially slashes — preserves ownership and active tracking
    //
    // it preserves ownership and active tracking on partial slash
    // ---------------------------------------------------------------------------

    function test_WhenTrusteePartiallySlashesPreservesOwnershipAndActiveTracking() external {
        // Alice deposits 5 ether
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Trustee slashes 50% of the deposit (2.5 ether)
        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 2.5 ether, 0, TEST_URL, TEST_SHA256);

        // it preserves ownership after partial slash
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), ALICE);

        // it keeps node in active tracking
        assertEq(collateral.getActiveNodeCount(), 1);

        // it keeps node in collaterals list
        assertEq(_allCollaterals().length, 1);

        // it leaves correct remaining balance
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 2.5 ether);
    }

    // ---------------------------------------------------------------------------
    // Test: when trustee denies reclaim with non-zero remaining balance
    //
    // it preserves nodeToMiner after deny with balance
    // it reverts when another miner tries to deposit after deny
    // it allows original miner to re-deposit and reclaim after deny
    // ---------------------------------------------------------------------------

    function test_WhenTrusteeDeniesReclaimWithNon_zeroRemainingBalance() external {
        // Alice deposits 5 ether
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Alice initiates reclaim (reclaimId=0)
        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        // Trustee denies — no slash, so taoCollaterals stays at 5 ether
        vm.prank(TRUSTEE);
        collateral.denyReclaimRequest(0, TEST_URL, TEST_SHA256);

        // it preserves nodeToMiner after deny with non-zero remaining balance
        // (deny only clears ownership when ALL balances become zero)
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), ALICE);
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 5 ether);

        // it reverts when another miner tries to deposit after deny (Alice still owns)
        vm.prank(BOB);
        vm.expectRevert(abi.encodeWithSelector(CollateralUpgradeable.NodeNotOwned.selector));
        collateral.deposit{value: 3 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // it allows original miner to re-deposit and reclaim after deny
        // Alice deposits 3 more ether (total 8 ether)
        vm.prank(ALICE);
        collateral.deposit{value: 3 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 8 ether);

        // Alice reclaims again (reclaimId=1, since reclaimId=0 was denied/deleted)
        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        // Warp past timeout and finalize — Alice should receive 8 ether
        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);
        uint256 aliceBalanceBefore = ALICE.balance;
        collateral.finalizeReclaim(1);

        assertEq(ALICE.balance, aliceBalanceBefore + 8 ether);
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 0);
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), address(0));
    }

    // ---------------------------------------------------------------------------
    // Test: when removing the last element via swap-and-pop
    //
    // Setup: A(HOTKEY_1), B(HOTKEY_2), C(HOTKEY_3) — slash C (last element)
    //
    // it decrements active node count
    // it preserves ordering of remaining elements without swap
    // it allows re-claiming the vacated node slot
    // ---------------------------------------------------------------------------

    function test_WhenRemovingTheLastElementViaSwap_and_pop() external {
        // Deposit A: Alice on HOTKEY_1 (pre-seeded by setUp)
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Deposit B: Bob on HOTKEY_2 (first claim — needs tx.origin == msg.sender)
        vm.prank(BOB, BOB);
        collateral.deposit{value: 3 ether}(HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 0);

        // Deposit C: Charlie on HOTKEY_3 (seed first, then deposit)
        _seedNodeOwner(HOTKEY_3, EXECUTOR_ID_3, CHARLIE);
        vm.prank(CHARLIE);
        collateral.deposit{value: 2 ether}(HOTKEY_3, EXECUTOR_ID_3, ALPHA_HOTKEY, 0);

        assertEq(collateral.getActiveNodeCount(), 3);

        // Slash C fully — C is the last element in activeNodeKeys, so just pop (no swap)
        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_3, EXECUTOR_ID_3, 2 ether, 0, TEST_URL, TEST_SHA256);

        // it decrements active node count
        assertEq(collateral.getActiveNodeCount(), 2);

        // it preserves ordering of remaining elements without swap
        // A should remain at index 0, B at index 1
        CollateralUpgradeable.NodeCollateral[] memory results = _allCollaterals();
        assertEq(results.length, 2);
        assertEq(results[0].minerHotkey, HOTKEY_1);
        assertEq(results[0].nodeId, EXECUTOR_ID_1);
        assertEq(results[1].minerHotkey, HOTKEY_2);
        assertEq(results[1].nodeId, EXECUTOR_ID_2);

        // it allows re-claiming the vacated node slot
        // Charlie is the new claimant — nodeToMiner is cleared, so needs tx.origin check
        vm.prank(CHARLIE, CHARLIE);
        collateral.deposit{value: 2 ether}(HOTKEY_3, EXECUTOR_ID_3, ALPHA_HOTKEY, 0);

        assertEq(collateral.nodeToMiner(HOTKEY_3, EXECUTOR_ID_3), CHARLIE);
        assertEq(collateral.getActiveNodeCount(), 3);
    }

    // ---------------------------------------------------------------------------
    // Test: when removing the first element via swap-and-pop
    //
    // Setup: A(HOTKEY_1), B(HOTKEY_2), C(HOTKEY_3) — slash A (first element → C swaps to slot 0)
    //
    // it decrements active node count
    // it moves last element into first position
    // it preserves remaining nodes collateral data after swap
    // ---------------------------------------------------------------------------

    function test_WhenRemovingTheFirstElementViaSwap_and_pop() external {
        // Deposit A: Alice on HOTKEY_1 (pre-seeded by setUp)
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Deposit B: Bob on HOTKEY_2
        vm.prank(BOB, BOB);
        collateral.deposit{value: 3 ether}(HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 0);

        // Deposit C: Charlie on HOTKEY_3
        _seedNodeOwner(HOTKEY_3, EXECUTOR_ID_3, CHARLIE);
        vm.prank(CHARLIE);
        collateral.deposit{value: 2 ether}(HOTKEY_3, EXECUTOR_ID_3, ALPHA_HOTKEY, 0);

        assertEq(collateral.getActiveNodeCount(), 3);
        // activeNodeKeys = [A, B, C] (1-based indices: A=1, B=2, C=3)

        // Slash A fully — A is at index 0, C (last) swaps into slot 0
        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 5 ether, 0, TEST_URL, TEST_SHA256);

        // it decrements active node count
        assertEq(collateral.getActiveNodeCount(), 2);

        CollateralUpgradeable.NodeCollateral[] memory results = _allCollaterals();
        assertEq(results.length, 2);

        // it moves last element (C) into first position
        assertEq(results[0].minerHotkey, HOTKEY_3);
        assertEq(results[0].nodeId, EXECUTOR_ID_3);

        // it preserves remaining nodes collateral data after swap
        // B is still at index 1 with its original collateral
        assertEq(results[1].minerHotkey, HOTKEY_2);
        assertEq(results[1].nodeId, EXECUTOR_ID_2);
        assertEq(results[0].taoCollateral, 2 ether); // C's balance unchanged
        assertEq(results[1].taoCollateral, 3 ether); // B's balance unchanged
    }

    // ---------------------------------------------------------------------------
    // Test: when multiple consecutive removals occur via swap-and-pop
    //
    // Setup: A(HOTKEY_1), B(HOTKEY_2), C(HOTKEY_3), D(HOTKEY_4)
    // Step 1: slash B → D swaps to slot 1 → [A, D, C]
    // Step 2: slash D → C swaps to slot 1 → [A, C]
    // Step 3: slash A → [C]
    //
    // it maintains correct active set after first removal
    // it maintains correct active set after second removal
    // it allows continued operations on remaining nodes
    // ---------------------------------------------------------------------------

    function test_WhenMultipleConsecutiveRemovalsOccurViaSwap_and_pop() external {
        // Deposit A: Alice on HOTKEY_1 (pre-seeded by setUp)
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Deposit B: Bob on HOTKEY_2
        vm.prank(BOB, BOB);
        collateral.deposit{value: 3 ether}(HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 0);

        // Deposit C: Charlie on HOTKEY_3
        _seedNodeOwner(HOTKEY_3, EXECUTOR_ID_3, CHARLIE);
        vm.prank(CHARLIE);
        collateral.deposit{value: 2 ether}(HOTKEY_3, EXECUTOR_ID_3, ALPHA_HOTKEY, 0);

        // Deposit D: Dave on HOTKEY_4 (first claim)
        vm.prank(DAVE, DAVE);
        collateral.deposit{value: 4 ether}(HOTKEY_4, EXECUTOR_ID_4, ALPHA_HOTKEY, 0);

        assertEq(collateral.getActiveNodeCount(), 4);
        // activeNodeKeys = [A, B, C, D] (indices: A=1, B=2, C=3, D=4)

        // Step 1: Slash B fully — D (last) swaps to B's slot → [A, D, C]
        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_2, EXECUTOR_ID_2, 3 ether, 0, TEST_URL, TEST_SHA256);

        // it maintains correct active set after first removal
        assertEq(collateral.getActiveNodeCount(), 3);
        {
            CollateralUpgradeable.NodeCollateral[] memory results = _allCollaterals();
            assertEq(results.length, 3);
            assertEq(results[0].minerHotkey, HOTKEY_1); // A at slot 0
            assertEq(results[1].minerHotkey, HOTKEY_4); // D swapped to slot 1
            assertEq(results[2].minerHotkey, HOTKEY_3); // C at slot 2
        }

        // Step 2: Slash D fully — C (last) swaps to D's slot → [A, C]
        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_4, EXECUTOR_ID_4, 4 ether, 0, TEST_URL, TEST_SHA256);

        // it maintains correct active set after second removal
        assertEq(collateral.getActiveNodeCount(), 2);
        {
            CollateralUpgradeable.NodeCollateral[] memory results = _allCollaterals();
            assertEq(results.length, 2);
            assertEq(results[0].minerHotkey, HOTKEY_1); // A at slot 0
            assertEq(results[1].minerHotkey, HOTKEY_3); // C swapped to slot 1
        }

        // Step 3: Slash A fully — only C remains
        vm.prank(TRUSTEE);
        collateral.slashCollateral(HOTKEY_1, EXECUTOR_ID_1, 5 ether, 0, TEST_URL, TEST_SHA256);

        // it allows continued operations on remaining nodes
        assertEq(collateral.getActiveNodeCount(), 1);
        {
            CollateralUpgradeable.NodeCollateral[] memory results = _allCollaterals();
            assertEq(results.length, 1);
            assertEq(results[0].minerHotkey, HOTKEY_3);
            assertEq(results[0].nodeId, EXECUTOR_ID_3);
            assertEq(results[0].miner, CHARLIE);
            assertEq(results[0].taoCollateral, 2 ether);
        }
    }

    // ---------------------------------------------------------------------------
    // Test: when removing a reclaim from the middle of the active reclaim set
    //
    // Setup: Node A (ALICE/HOTKEY_1), Node B (BOB/HOTKEY_2), Node C (CHARLIE/HOTKEY_3)
    // Each deposits and reclaims → reclaimIds [0, 1, 2]
    // Deny R1 → last (R2) swaps into slot 1 → [0, 2]
    //
    // it moves last reclaim id into the vacated slot
    // it preserves correct active reclaim set ordering
    // ---------------------------------------------------------------------------

    function test_WhenRemovingAReclaimFromTheMiddleOfTheActiveReclaimSet() external {
        // Node A: Alice on HOTKEY_1 (pre-seeded by setUp)
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);
        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);
        // reclaimId = 0

        // Node B: Bob on HOTKEY_2 (first claim)
        vm.prank(BOB, BOB);
        collateral.deposit{value: 3 ether}(HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 0);
        vm.prank(BOB);
        collateral.reclaimCollateral(HOTKEY_2, EXECUTOR_ID_2);
        // reclaimId = 1

        // Node C: Charlie on HOTKEY_3 (first claim)
        vm.prank(CHARLIE, CHARLIE);
        collateral.deposit{value: 2 ether}(HOTKEY_3, EXECUTOR_ID_3, ALPHA_HOTKEY, 0);
        vm.prank(CHARLIE);
        collateral.reclaimCollateral(HOTKEY_3, EXECUTOR_ID_3);
        // reclaimId = 2

        // activeReclaimIds = [0, 1, 2]
        assertEq(collateral.getActiveReclaimCount(), 3);

        // Deny R1 (middle): R2 (last) swaps into slot 1 → [0, 2]
        vm.prank(TRUSTEE);
        collateral.denyReclaimRequest(1, TEST_URL, TEST_SHA256);

        // it moves last reclaim id into the vacated slot
        assertEq(collateral.getActiveReclaimCount(), 2);

        // it preserves correct active reclaim set ordering
        CollateralUpgradeable.ReclaimInfo[] memory results = _allReclaims();
        assertEq(results.length, 2);
        assertEq(results[0].reclaimRequestId, 0);
        assertEq(results[0].minerHotkey, HOTKEY_1);
        assertEq(results[1].reclaimRequestId, 2);
        assertEq(results[1].minerHotkey, HOTKEY_3);
    }

    // ---------------------------------------------------------------------------
    // Test: when removing the first reclaim via swap-and-pop
    //
    // Same 3-node/3-reclaim setup. Finalize R0 (first) → R2 (last) swaps to slot 0 → [2, 1]
    //
    // it moves last reclaim id into first position
    // it preserves remaining reclaims after swap
    // ---------------------------------------------------------------------------

    function test_WhenRemovingTheFirstReclaimViaSwap_and_pop() external {
        // Node A: Alice on HOTKEY_1 (pre-seeded by setUp)
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);
        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);
        // reclaimId = 0

        // Node B: Bob on HOTKEY_2 (first claim)
        vm.prank(BOB, BOB);
        collateral.deposit{value: 3 ether}(HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 0);
        vm.prank(BOB);
        collateral.reclaimCollateral(HOTKEY_2, EXECUTOR_ID_2);
        // reclaimId = 1

        // Node C: Charlie on HOTKEY_3 (first claim)
        vm.prank(CHARLIE, CHARLIE);
        collateral.deposit{value: 2 ether}(HOTKEY_3, EXECUTOR_ID_3, ALPHA_HOTKEY, 0);
        vm.prank(CHARLIE);
        collateral.reclaimCollateral(HOTKEY_3, EXECUTOR_ID_3);
        // reclaimId = 2

        // activeReclaimIds = [0, 1, 2]
        assertEq(collateral.getActiveReclaimCount(), 3);

        // Warp past DECISION_TIMEOUT and finalize R0 (first) → R2 swaps to slot 0 → [2, 1]
        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);
        uint256 aliceBalanceBefore = ALICE.balance;
        collateral.finalizeReclaim(0);

        // it moves last reclaim id into first position
        assertEq(collateral.getActiveReclaimCount(), 2);

        CollateralUpgradeable.ReclaimInfo[] memory results = _allReclaims();
        assertEq(results.length, 2);
        assertEq(results[0].reclaimRequestId, 2);
        assertEq(results[0].minerHotkey, HOTKEY_3);
        assertEq(results[1].reclaimRequestId, 1);
        assertEq(results[1].minerHotkey, HOTKEY_2);

        // it preserves remaining reclaims after swap: Alice received her 5 ether back
        assertEq(ALICE.balance, aliceBalanceBefore + 5 ether);
    }

    // ---------------------------------------------------------------------------
    // Test: when multiple consecutive reclaim removals occur
    //
    // Same 3-node/3-reclaim setup.
    // Deny R1 → [0, 2], count == 2
    // Deny R0 → R2 swaps to slot 0 → [2], count == 1
    //
    // it maintains correct active set after first reclaim removal
    // it maintains correct active set after second reclaim removal
    // ---------------------------------------------------------------------------

    function test_WhenMultipleConsecutiveReclaimRemovalsOccur() external {
        // Node A: Alice on HOTKEY_1 (pre-seeded by setUp)
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);
        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);
        // reclaimId = 0

        // Node B: Bob on HOTKEY_2 (first claim)
        vm.prank(BOB, BOB);
        collateral.deposit{value: 3 ether}(HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 0);
        vm.prank(BOB);
        collateral.reclaimCollateral(HOTKEY_2, EXECUTOR_ID_2);
        // reclaimId = 1

        // Node C: Charlie on HOTKEY_3 (first claim)
        vm.prank(CHARLIE, CHARLIE);
        collateral.deposit{value: 2 ether}(HOTKEY_3, EXECUTOR_ID_3, ALPHA_HOTKEY, 0);
        vm.prank(CHARLIE);
        collateral.reclaimCollateral(HOTKEY_3, EXECUTOR_ID_3);
        // reclaimId = 2

        // activeReclaimIds = [0, 1, 2]
        assertEq(collateral.getActiveReclaimCount(), 3);

        // First removal: deny R1 → R2 swaps to slot 1 → [0, 2]
        vm.prank(TRUSTEE);
        collateral.denyReclaimRequest(1, TEST_URL, TEST_SHA256);

        // it maintains correct active set after first reclaim removal
        assertEq(collateral.getActiveReclaimCount(), 2);
        {
            CollateralUpgradeable.ReclaimInfo[] memory results = _allReclaims();
            assertEq(results.length, 2);
            assertEq(results[0].reclaimRequestId, 0);
            assertEq(results[1].reclaimRequestId, 2);
        }

        // Second removal: deny R0 → R2 (now at slot 1, last) swaps to slot 0 → [2]
        vm.prank(TRUSTEE);
        collateral.denyReclaimRequest(0, TEST_URL, TEST_SHA256);

        // it maintains correct active set after second reclaim removal
        assertEq(collateral.getActiveReclaimCount(), 1);
        {
            CollateralUpgradeable.ReclaimInfo[] memory results = _allReclaims();
            assertEq(results.length, 1);
            assertEq(results[0].reclaimRequestId, 2);
            assertEq(results[0].minerHotkey, HOTKEY_3);
        }
    }

    // ---------------------------------------------------------------------------
    // Test: when miner re-deposits during an active reclaim
    //
    // it does not include re-deposit amount in the original reclaim
    // it preserves ownership after original reclaim finalizes
    // it allows subsequent reclaim of the extra deposit
    // ---------------------------------------------------------------------------

    function test_WhenMinerRe_depositsDuringAnActiveReclaim() external {
        // Alice deposits 10 ether (HOTKEY_1 pre-seeded to ALICE in setUp)
        vm.prank(ALICE);
        collateral.deposit{value: 10 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        // Alice initiates reclaim → reclaimId=0, pending=10 ether
        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);

        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 10 ether);

        // Alice re-deposits 3 more ether while reclaim is in-flight
        // (she still owns the node so this succeeds)
        vm.prank(ALICE);
        collateral.deposit{value: 3 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);

        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 13 ether);

        // Warp past timeout and finalize the original reclaim
        // actualAmount = min(pendingAmount=10, taoCollaterals=13) = 10
        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);
        uint256 aliceBalanceBefore = ALICE.balance;
        collateral.finalizeReclaim(0);

        // it does not include re-deposit amount in the original reclaim
        assertEq(ALICE.balance, aliceBalanceBefore + 10 ether);

        // it preserves ownership after original reclaim finalizes
        // 3 ether remains, so nodeToMiner is NOT cleared
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), ALICE);
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 3 ether);

        // it allows subsequent reclaim of the extra deposit
        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);
        // reclaimId = 1

        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);
        uint256 aliceBalanceBefore2 = ALICE.balance;
        collateral.finalizeReclaim(1);

        assertEq(ALICE.balance, aliceBalanceBefore2 + 3 ether);
        assertEq(collateral.taoCollaterals(HOTKEY_1, EXECUTOR_ID_1), 0);
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), address(0));
    }

    // ---------------------------------------------------------------------------
    // Test: when a previously swapped reclaim is subsequently removed
    //
    // Setup: 4 nodes (HOTKEY_1–4), 4 reclaims → [R0, R1, R2, R3]
    // Step 1: Deny R1 (middle) → R3 swaps to slot 1 → [R0, R3, R2]
    //         activeReclaimIdIndex[R3] must be updated from 4 → 2
    // Step 2: Deny R3 (the just-swapped item) → R2 swaps to slot 1 → [R0, R2]
    //         This reads activeReclaimIdIndex[R3]; if not updated (still 4),
    //         the access would be out-of-bounds/corrupt
    //
    // it uses the updated index to locate and remove the swapped reclaim
    // it produces a correct active set after removing the swapped reclaim
    // ---------------------------------------------------------------------------

    function test_WhenAPreviouslySwappedReclaimIsSubsequentlyRemoved() external {
        // Node A: Alice on HOTKEY_1 (pre-seeded by setUp)
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(HOTKEY_1, EXECUTOR_ID_1, ALPHA_HOTKEY, 0);
        vm.prank(ALICE);
        collateral.reclaimCollateral(HOTKEY_1, EXECUTOR_ID_1);
        // reclaimId = 0 (R0)

        // Node B: Bob on HOTKEY_2 (first claim)
        vm.prank(BOB, BOB);
        collateral.deposit{value: 3 ether}(HOTKEY_2, EXECUTOR_ID_2, ALPHA_HOTKEY, 0);
        vm.prank(BOB);
        collateral.reclaimCollateral(HOTKEY_2, EXECUTOR_ID_2);
        // reclaimId = 1 (R1)

        // Node C: Charlie on HOTKEY_3 (first claim)
        vm.prank(CHARLIE, CHARLIE);
        collateral.deposit{value: 2 ether}(HOTKEY_3, EXECUTOR_ID_3, ALPHA_HOTKEY, 0);
        vm.prank(CHARLIE);
        collateral.reclaimCollateral(HOTKEY_3, EXECUTOR_ID_3);
        // reclaimId = 2 (R2)

        // Node D: Dave on HOTKEY_4 (first claim)
        vm.prank(DAVE, DAVE);
        collateral.deposit{value: 4 ether}(HOTKEY_4, EXECUTOR_ID_4, ALPHA_HOTKEY, 0);
        vm.prank(DAVE);
        collateral.reclaimCollateral(HOTKEY_4, EXECUTOR_ID_4);
        // reclaimId = 3 (R3)

        // activeReclaimIds = [R0, R1, R2, R3]; activeReclaimIdIndex[R3] = 4
        assertEq(collateral.getActiveReclaimCount(), 4);

        // Step 1: Deny R1 (middle) → R3 (last) swaps to slot 1 → [R0, R3, R2]
        // This must update activeReclaimIdIndex[R3] from 4 → 2
        vm.prank(TRUSTEE);
        collateral.denyReclaimRequest(1, TEST_URL, TEST_SHA256);

        assertEq(collateral.getActiveReclaimCount(), 3);

        // Step 2: Deny R3 (the just-swapped item)
        // Contract reads activeReclaimIdIndex[R3]; if still 4 → out-of-bounds/corrupt
        // With correct index (2), finds R3 at slot 1, removes it
        // R2 (last, slot 2) swaps to slot 1 → activeReclaimIds = [R0, R2]
        vm.prank(TRUSTEE);
        collateral.denyReclaimRequest(3, TEST_URL, TEST_SHA256);

        // it uses the updated index to locate and remove the swapped reclaim
        // it produces a correct active set after removing the swapped reclaim
        assertEq(collateral.getActiveReclaimCount(), 2);

        CollateralUpgradeable.ReclaimInfo[] memory results = _allReclaims();
        assertEq(results.length, 2);
        assertEq(results[0].reclaimRequestId, 0); // R0 still at slot 0
        assertEq(results[1].reclaimRequestId, 2); // R2 swapped to slot 1
    }

    // ---------------------------------------------------------------------------
    // Events (re-declared from CollateralUpgradeable for vm.expectEmit)
    // ---------------------------------------------------------------------------

    event Denied(uint256 indexed reclaimRequestId, string url, bytes32 urlContentSha256);
    event Slashed(
        bytes32 indexed minerHotkey,
        bytes16 indexed executorId,
        address indexed miner,
        uint256 slashAmount,
        uint256 slashAlphaAmount,
        string url,
        bytes32 urlContentSha256
    );
    event Reclaimed(
        uint256 indexed reclaimRequestId,
        bytes32 indexed minerHotkey,
        bytes16 indexed executorId,
        address miner,
        uint256 amount,
        bytes32 alphaColdkey,
        uint256 alphaAmount
    );
}

// ---------------------------------------------------------------------------
// Precompile mocks — defined after the test contract so bulloak finds
// NodeOwnershipEdgeCasesTest as the first contract in the file.
// ---------------------------------------------------------------------------

contract _AddressMappingMock {
    function addressMapping(address evmAddress) external pure returns (bytes32) {
        return bytes32(uint256(uint160(evmAddress)));
    }
}

contract _StakingV2Mock {
    function transferStake(bytes32, bytes32, uint256, uint256, uint256) external payable {}
    function moveStake(bytes32, bytes32, uint256, uint256, uint256) external payable {}
    function getStake(bytes32, bytes32, uint256) external view returns (uint256) {
        return type(uint256).max - gasleft();
    }
}
