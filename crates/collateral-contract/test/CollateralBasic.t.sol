// SPDX-License-Identifier: UNLICENSED

pragma solidity ^0.8.22;
import {Test, console} from "forge-std/Test.sol";
import {CollateralUpgradeable} from "../src/CollateralUpgradeable.sol";
import {CollateralUpgradeableV2} from "../src/CollateralUpgradeableV2.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";

/**
 * @title CollateralBasicTest
 * @notice Basic tests for CollateralUpgradeable without alpha/IStaking interactions
 * @dev These tests focus on core functionality that doesn't require IStaking mocking
 */
contract CollateralBasicTest is Test {
    CollateralUpgradeable public collateral;
    CollateralUpgradeable public implementation;
    ERC1967Proxy public proxy;

    // Test parameters
    uint16 constant NETUID = 42;
    address constant TRUSTEE = address(0x123);
    uint256 constant MIN_DEPOSIT = 1 ether;
    uint64 constant DECISION_TIMEOUT = 3600; // 1 hour
    address constant ADMIN = address(0x456);
    address constant ALICE = address(0x789);
    address constant BOB = address(0xABC);
    address constant CHARLIE = address(0xDEF);

    bytes32 constant ALPHA_HOTKEY = bytes32(uint256(1));
    bytes32 constant CONTRACT_COLDKEY = bytes32(uint256(2));
    bytes32 constant HOTKEY_1 = bytes32(uint256(100));
    bytes32 constant HOTKEY_2 = bytes32(uint256(101));
    bytes16 constant EXECUTOR_ID_1 = bytes16(uint128(1));
    bytes16 constant EXECUTOR_ID_2 = bytes16(uint128(2));

    string constant TEST_URL = "https://example.com/proof";
    bytes16 constant TEST_MD5 =
        bytes16(uint128(0x12345678901234567890123456789012));

    function setUp() public {
        // Deploy implementation
        implementation = new CollateralUpgradeable();

        // Prepare initialization data
        bytes memory initData = abi.encodeWithSelector(
            CollateralUpgradeable.initialize.selector,
            NETUID,
            TRUSTEE,
            MIN_DEPOSIT,
            DECISION_TIMEOUT,
            ADMIN,
            ALPHA_HOTKEY
        );

        // Deploy proxy
        proxy = new ERC1967Proxy(address(implementation), initData);

        // Cast proxy to interface
        collateral = CollateralUpgradeable(payable(address(proxy)));

        // Set contract coldkey
        vm.prank(TRUSTEE);
        collateral.setContractColdkey(CONTRACT_COLDKEY);

        // Give test accounts some ETH
        vm.deal(ALICE, 100 ether);
        vm.deal(BOB, 100 ether);
        vm.deal(CHARLIE, 100 ether);
    }

    // ============ INITIALIZATION TESTS ============

    function testInitialization() public view {
        assertEq(collateral.NETUID(), NETUID);
        assertEq(collateral.TRUSTEE(), TRUSTEE);
        assertEq(collateral.MIN_COLLATERAL_INCREASE(), MIN_DEPOSIT);
        assertEq(collateral.DECISION_TIMEOUT(), DECISION_TIMEOUT);
        assertEq(collateral.getVersion(), 1);
        assertEq(collateral.CONTRACT_COLDKEY(), CONTRACT_COLDKEY);
        assertEq(collateral.CONTRACT_HOTKEY(), ALPHA_HOTKEY);

        // Check roles
        assertTrue(collateral.hasRole(collateral.DEFAULT_ADMIN_ROLE(), ADMIN));
        assertTrue(collateral.hasRole(collateral.UPGRADER_ROLE(), ADMIN));
    }

    function testCannotInitializeTwice() public {
        vm.expectRevert();
        collateral.initialize(
            NETUID,
            TRUSTEE,
            MIN_DEPOSIT,
            DECISION_TIMEOUT,
            ADMIN,
            ALPHA_HOTKEY
        );
    }

    // ============ DEPOSIT TESTS (WITHOUT ALPHA) ============

    function testDepositInsufficientAmount() public {
        vm.prank(ALICE);
        vm.expectRevert(
            abi.encodeWithSelector(
                CollateralUpgradeable.InsufficientAmount.selector
            )
        );
        collateral.deposit{value: 0.5 ether}(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALPHA_HOTKEY,
            0
        );
    }

    function testDepositExecutorNotOwned() public {
        // Alice makes first deposit
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALPHA_HOTKEY,
            0
        );

        // Bob tries to deposit to same executor
        vm.prank(BOB);
        vm.expectRevert(
            abi.encodeWithSelector(CollateralUpgradeable.NodeNotOwned.selector)
        );
        collateral.deposit{value: 5 ether}(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALPHA_HOTKEY,
            0
        );
    }

    function testMultipleDepositsFromSameOwner() public {
        // First deposit
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALPHA_HOTKEY,
            0
        );

        // Second deposit from same owner
        vm.prank(ALICE);
        collateral.deposit{value: 3 ether}(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALPHA_HOTKEY,
            0
        );

        assertEq(collateral.collaterals(HOTKEY_1, EXECUTOR_ID_1), 8 ether);
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), ALICE);
    }

    // ============ RECLAIM TESTS ============

    function testReclaimCollateral() public {
        // Setup: Alice deposits
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALPHA_HOTKEY,
            0
        );

        // Alice initiates reclaim
        vm.expectEmit(true, true, true, true, address(collateral));
        emit ReclaimProcessStarted(
            0,
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALICE,
            5 ether,
            bytes32(0),
            0,
            uint64(block.timestamp + DECISION_TIMEOUT),
            TEST_URL,
            TEST_MD5
        );

        vm.prank(ALICE);
        collateral.reclaimCollateral(
            HOTKEY_1,
            EXECUTOR_ID_1,
            bytes32(0),
            TEST_URL,
            TEST_MD5
        );

        // Check reclaim was created
        (
            bytes32 hotkey,
            bytes16 executorId,
            address miner,
            uint256 amount,
            ,
            ,
            uint64 denyTimeout
        ) = collateral.reclaims(0);
        assertEq(hotkey, HOTKEY_1);
        assertEq(executorId, EXECUTOR_ID_1);
        assertEq(miner, ALICE);
        assertEq(amount, 5 ether);
        assertEq(denyTimeout, block.timestamp + DECISION_TIMEOUT);
    }

    function testReclaimExecutorNotOwned() public {
        // Alice deposits
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALPHA_HOTKEY,
            0
        );

        // Bob tries to reclaim
        vm.prank(BOB);
        vm.expectRevert(
            abi.encodeWithSelector(CollateralUpgradeable.NodeNotOwned.selector)
        );
        collateral.reclaimCollateral(
            HOTKEY_1,
            EXECUTOR_ID_1,
            bytes32(0),
            TEST_URL,
            TEST_MD5
        );
    }

    function testReclaimAmountZero() public {
        // Try to reclaim without any deposits
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALPHA_HOTKEY,
            0
        );

        vm.prank(ALICE);
        collateral.reclaimCollateral(
            HOTKEY_1,
            EXECUTOR_ID_1,
            bytes32(0),
            TEST_URL,
            TEST_MD5
        );

        vm.prank(ALICE);
        vm.expectRevert(
            abi.encodeWithSelector(CollateralUpgradeable.AmountZero.selector)
        );
        collateral.reclaimCollateral(
            HOTKEY_1,
            EXECUTOR_ID_1,
            bytes32(0),
            TEST_URL,
            TEST_MD5
        );
    }

    // ============ FINALIZE RECLAIM TESTS ============

    function testFinalizeReclaim() public {
        // Setup: Alice deposits and initiates reclaim
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALPHA_HOTKEY,
            0
        );

        vm.prank(ALICE);
        collateral.reclaimCollateral(
            HOTKEY_1,
            EXECUTOR_ID_1,
            bytes32(0),
            TEST_URL,
            TEST_MD5
        );

        // Fast forward past timeout
        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);

        uint256 aliceBalanceBefore = ALICE.balance;

        // Finalize reclaim
        vm.expectEmit(true, true, true, true, address(collateral));
        emit Reclaimed(
            0,
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALICE,
            5 ether,
            bytes32(0),
            0
        );

        collateral.finalizeReclaim(0);

        // Check state
        assertEq(ALICE.balance, aliceBalanceBefore + 5 ether);
        assertEq(collateral.collaterals(HOTKEY_1, EXECUTOR_ID_1), 0);
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), address(0));
    }

    function testFinalizeReclaimBeforeTimeout() public {
        // Setup reclaim
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALPHA_HOTKEY,
            0
        );

        vm.prank(ALICE);
        collateral.reclaimCollateral(
            HOTKEY_1,
            EXECUTOR_ID_1,
            bytes32(0),
            TEST_URL,
            TEST_MD5
        );

        // Try to finalize before timeout
        vm.expectRevert(
            abi.encodeWithSelector(
                CollateralUpgradeable.BeforeDenyTimeout.selector
            )
        );
        collateral.finalizeReclaim(0);
    }

    function testFinalizeReclaimNotFound() public {
        vm.expectRevert(
            abi.encodeWithSelector(
                CollateralUpgradeable.ReclaimNotFound.selector
            )
        );
        collateral.finalizeReclaim(999);
    }

    function testFinalizeReclaimInsufficientCollateral() public {
        // Setup reclaim
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALPHA_HOTKEY,
            0
        );

        vm.prank(ALICE);
        collateral.reclaimCollateral(
            HOTKEY_1,
            EXECUTOR_ID_1,
            bytes32(0),
            TEST_URL,
            TEST_MD5
        );

        // Slash some collateral
        vm.prank(TRUSTEE);
        collateral.slashCollateral(
            HOTKEY_1,
            EXECUTOR_ID_1,
            3 ether,
            0,
            TEST_URL,
            TEST_MD5
        );

        // Fast forward and try to finalize
        vm.warp(block.timestamp + DECISION_TIMEOUT + 1);

        vm.expectRevert(
            abi.encodeWithSelector(
                CollateralUpgradeable.InsufficientCollateralForReclaim.selector
            )
        );
        collateral.finalizeReclaim(0);
    }

    // ============ DENY RECLAIM TESTS ============

    function testDenyReclaimRequest() public {
        // Setup reclaim
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALPHA_HOTKEY,
            0
        );

        vm.prank(ALICE);
        collateral.reclaimCollateral(
            HOTKEY_1,
            EXECUTOR_ID_1,
            bytes32(0),
            TEST_URL,
            TEST_MD5
        );

        // Trustee denies
        vm.expectEmit(true, false, false, true, address(collateral));
        emit Denied(0, TEST_URL, TEST_MD5);

        uint256 amount;
        (, , , amount, , , ) = collateral.reclaims(0);
        assertEq(amount, 5 ether);

        vm.prank(TRUSTEE);
        collateral.denyReclaimRequest(0, TEST_URL, TEST_MD5);

        // Check reclaim was deleted
        (, , , amount, , , ) = collateral.reclaims(0);
        assertEq(amount, 0);
    }

    function testDenyReclaimNotTrustee() public {
        // Setup reclaim
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALPHA_HOTKEY,
            0
        );

        vm.prank(ALICE);
        collateral.reclaimCollateral(
            HOTKEY_1,
            EXECUTOR_ID_1,
            bytes32(0),
            TEST_URL,
            TEST_MD5
        );

        // Non-trustee tries to deny
        vm.prank(BOB);
        vm.expectRevert(
            abi.encodeWithSelector(CollateralUpgradeable.NotTrustee.selector)
        );
        collateral.denyReclaimRequest(1, TEST_URL, TEST_MD5);
    }

    // ============ SLASH TESTS ============

    function testSlashCollateral() public {
        // Setup
        vm.prank(ALICE);
        collateral.deposit{value: 10 ether}(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALPHA_HOTKEY,
            0
        );

        uint256 contractBalanceBefore = address(collateral).balance;

        // Slash partial amount
        vm.expectEmit(true, true, true, true, address(collateral));
        emit Slashed(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALICE,
            5 ether,
            0,
            TEST_URL,
            TEST_MD5
        );

        vm.prank(TRUSTEE);
        collateral.slashCollateral(
            HOTKEY_1,
            EXECUTOR_ID_1,
            5 ether,
            0,
            TEST_URL,
            TEST_MD5
        );

        // Check state
        assertEq(collateral.collaterals(HOTKEY_1, EXECUTOR_ID_1), 5 ether);
        assertEq(address(collateral).balance, contractBalanceBefore - 5 ether);
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), ALICE); // Still owned
    }

    function testSlashAllCollateral() public {
        // Setup
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALPHA_HOTKEY,
            0
        );

        // Slash all
        vm.prank(TRUSTEE);
        collateral.slashCollateral(
            HOTKEY_1,
            EXECUTOR_ID_1,
            5 ether,
            0,
            TEST_URL,
            TEST_MD5
        );

        // Check executor ownership is cleared
        assertEq(collateral.nodeToMiner(HOTKEY_1, EXECUTOR_ID_1), address(0));
        assertEq(collateral.collaterals(HOTKEY_1, EXECUTOR_ID_1), 0);
    }

    function testSlashNotTrustee() public {
        vm.prank(ALICE);
        collateral.deposit{value: 5 ether}(
            HOTKEY_1,
            EXECUTOR_ID_1,
            ALPHA_HOTKEY,
            0
        );

        vm.prank(BOB);
        vm.expectRevert(
            abi.encodeWithSelector(CollateralUpgradeable.NotTrustee.selector)
        );
        collateral.slashCollateral(
            HOTKEY_1,
            EXECUTOR_ID_1,
            5 ether,
            0,
            TEST_URL,
            TEST_MD5
        );
    }

    function testSlashAmountZero() public {
        vm.prank(TRUSTEE);
        vm.expectRevert(
            abi.encodeWithSelector(CollateralUpgradeable.AmountZero.selector)
        );
        collateral.slashCollateral(
            HOTKEY_1,
            EXECUTOR_ID_1,
            0,
            0,
            TEST_URL,
            TEST_MD5
        );
    }

    // ============ ADMIN FUNCTION TESTS ============

    function testUpdateTrustee() public {
        address newTrustee = makeAddr("newTrustee");

        vm.expectEmit(true, true, false, false, address(collateral));
        emit TrusteeUpdated(TRUSTEE, newTrustee);

        vm.prank(ADMIN);
        collateral.updateTrustee(newTrustee);

        assertEq(collateral.TRUSTEE(), newTrustee);
    }

    function testUpdateDecisionTimeout() public {
        vm.prank(ADMIN);
        collateral.updateDecisionTimeout(7200);

        assertEq(collateral.DECISION_TIMEOUT(), 7200);
    }

    function testUpdateMinCollateralIncrease() public {
        vm.prank(ADMIN);
        collateral.updateMinCollateralIncrease(2 ether);

        assertEq(collateral.MIN_COLLATERAL_INCREASE(), 2 ether);
    }

    function testSetContractColdkey() public {
        bytes32 newColdkey = bytes32(uint256(999));

        vm.expectEmit(true, true, false, false, address(collateral));
        emit AlphaColdkeyUpdated(CONTRACT_COLDKEY, newColdkey);

        vm.prank(TRUSTEE);
        collateral.setContractColdkey(newColdkey);

        assertEq(collateral.CONTRACT_COLDKEY(), newColdkey);
    }

    // ============ UPGRADE TESTS ============

    function testUpgrade() public {
        CollateralUpgradeableV2 newImplementation = new CollateralUpgradeableV2();

        vm.expectEmit(true, true, false, false, address(collateral));
        emit ContractUpgraded(2, address(newImplementation));

        vm.prank(ADMIN);
        collateral.upgradeToAndCall(address(newImplementation), "");

        assertEq(collateral.getVersion(), 2);
    }

    // ============ EVENTS ============

    event Deposit(
        bytes32 indexed hotkey,
        bytes16 indexed executorId,
        address indexed miner,
        uint256 amount,
        bytes32 alphaHotkey,
        uint256 alphaAmount
    );

    event ReclaimProcessStarted(
        uint256 indexed reclaimRequestId,
        bytes32 indexed hotkey,
        bytes16 indexed executorId,
        address miner,
        uint256 amount,
        bytes32 alphaColdkey,
        uint256 alphaAmount,
        uint64 expirationTime,
        string url,
        bytes16 urlContentMd5Checksum
    );

    event Reclaimed(
        uint256 indexed reclaimRequestId,
        bytes32 indexed hotkey,
        bytes16 indexed executorId,
        address miner,
        uint256 amount,
        bytes32 alphaColdkey,
        uint256 alphaAmount
    );

    event Denied(
        uint256 indexed reclaimRequestId,
        string url,
        bytes16 urlContentMd5Checksum
    );

    event Slashed(
        bytes32 indexed hotkey,
        bytes16 indexed executorId,
        address indexed miner,
        uint256 slashAmount,
        uint256 slashAlphaAmount,
        string url,
        bytes16 urlContentMd5Checksum
    );

    event AlphaColdkeyUpdated(
        bytes32 indexed oldAlphaColdkey,
        bytes32 indexed newAlphaColdkey
    );

    event ContractUpgraded(
        uint256 indexed newVersion,
        address indexed newImplementation
    );

    event TrusteeUpdated(
        address indexed oldTrustee,
        address indexed newTrustee
    );
}
