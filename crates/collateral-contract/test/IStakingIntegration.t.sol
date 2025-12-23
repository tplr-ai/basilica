// SPDX-License-Identifier: UNLICENSED

pragma solidity ^0.8.22;

import {Test, console} from "forge-std/Test.sol";
import {CollateralUpgradeable} from "../src/CollateralUpgradeable.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";

// Advanced Mock IStaking contract that simulates real behavior
contract AdvancedMockIStaking {
    // coldkey -> hotkey -> netuid -> amount
    mapping(bytes32 => mapping(bytes32 => mapping(uint256 => uint256)))
        public stakes;

    // Track transfer history for testing
    struct Transfer {
        bytes32 fromColdkey;
        bytes32 toColdkey;
        bytes32 fromHotkey;
        bytes32 toHotkey;
        uint256 fromNetuid;
        uint256 toNetuid;
        uint256 amount;
        uint256 timestamp;
    }

    Transfer[] public transferHistory;

    function transferStake(
        bytes32 toColdkey,
        bytes32 hotkey,
        uint256 fromNetuid,
        uint256 toNetuid,
        uint256 amount
    ) external {
        // directly convert address to coldkey
        bytes32 fromColdkey = bytes32(uint256(uint160(msg.sender)));
        require(
            stakes[fromColdkey][hotkey][fromNetuid] >= amount,
            "Insufficient stake"
        );
        stakes[fromColdkey][hotkey][fromNetuid] -= amount;
        stakes[toColdkey][hotkey][toNetuid] += amount;
        // Record transfer
        transferHistory.push(
            Transfer({
                fromColdkey: fromColdkey,
                toColdkey: toColdkey,
                fromHotkey: hotkey,
                toHotkey: hotkey,
                fromNetuid: fromNetuid,
                toNetuid: toNetuid,
                amount: amount,
                timestamp: block.timestamp
            })
        );
    }

    function moveStake(
        bytes32 fromHotkey,
        bytes32 toHotkey,
        uint256 fromNetuid,
        uint256 toNetuid,
        uint256 amount
    ) external {
        // directly convert address to coldkey
        bytes32 coldkey = bytes32(uint256(uint160(msg.sender)));
        require(
            stakes[coldkey][fromHotkey][fromNetuid] >= amount,
            "Insufficient stake"
        );

        stakes[coldkey][fromHotkey][fromNetuid] -= amount;
        stakes[coldkey][toHotkey][toNetuid] += amount;

        transferHistory.push(
            Transfer({
                fromColdkey: coldkey,
                toColdkey: coldkey,
                fromHotkey: fromHotkey,
                toHotkey: toHotkey,
                fromNetuid: fromNetuid,
                toNetuid: toNetuid,
                amount: amount,
                timestamp: block.timestamp
            })
        );
    }

    function getStake(
        bytes32 hotkey,
        bytes32 coldkey,
        uint256 netuid
    ) external view returns (uint256) {
        return stakes[coldkey][hotkey][netuid];
    }

    // Helper functions for testing
    function setStake(
        bytes32 hotkey,
        bytes32 coldkey,
        uint256 netuid,
        uint256 amount
    ) external {
        stakes[coldkey][hotkey][netuid] = amount;
    }

    function getTransferCount() external view returns (uint256) {
        return transferHistory.length;
    }

    function getLastTransfer() external view returns (Transfer memory) {
        require(transferHistory.length > 0, "No transfers");
        return transferHistory[transferHistory.length - 1];
    }
}

contract IStakingIntegrationTest is Test {
    mapping(bytes32 => mapping(bytes32 => mapping(uint256 => uint256)))
        public stakes;

    CollateralUpgradeable public collateral;
    AdvancedMockIStaking public mockStaking;

    // Test parameters
    uint16 constant NETUID = 39;

    uint256 constant MIN_DEPOSIT = 1 ether;
    uint64 constant DECISION_TIMEOUT = 3600;

    address constant ALICE = address(0x09);
    bytes32 constant ALICE_COLDKEY = bytes32(uint256(9));

    bytes32 CONTRACT_COLDKEY;
    bytes32 CONTRACT_HOTKEY = bytes32(uint256(88));

    bytes32 constant HOTKEY_1 = bytes32(uint256(101));
    bytes16 constant EXECUTOR_ID_1 = bytes16(uint128(1));

    address constant TRUSTEE = address(0x1111);
    address constant ADMIN = address(0x2222);

    uint256 constant ALPHA_AMOUNT = 5 ether;

    function setUp() public {
        // Deploy advanced mock staking
        mockStaking = new AdvancedMockIStaking();

        // Deploy collateral contract
        CollateralUpgradeable implementation = new CollateralUpgradeable();
        bytes memory initData = abi.encodeWithSelector(
            CollateralUpgradeable.initialize.selector,
            NETUID,
            TRUSTEE,
            MIN_DEPOSIT,
            DECISION_TIMEOUT,
            ADMIN,
            CONTRACT_HOTKEY
        );
        ERC1967Proxy proxy = new ERC1967Proxy(
            address(implementation),
            initData
        );
        collateral = CollateralUpgradeable(payable(address(proxy)));

        CONTRACT_COLDKEY = bytes32(uint256(uint160(address(proxy))));

        // Set contract coldkey
        vm.prank(TRUSTEE);
        collateral.setContractColdkey(CONTRACT_COLDKEY);

        // Mock the IStaking address
        vm.etch(
            0x0000000000000000000000000000000000000805,
            address(mockStaking).code
        );

        vm.deal(ALICE, 100 ether);
    }

    function testDepositWithAlphaTransfer() public {
        uint256 initialStake = mockStaking.getStake(
            CONTRACT_COLDKEY,
            CONTRACT_HOTKEY,
            NETUID
        );

        assertEq(initialStake, 0 ether);

        mockStaking.setStake(CONTRACT_HOTKEY, ALICE_COLDKEY, NETUID, 5 ether);

        uint256 finalStake = mockStaking.getStake(
            CONTRACT_HOTKEY,
            ALICE_COLDKEY,
            NETUID
        );

        assertEq(finalStake, 5 ether);

        vm.prank(ALICE);
        mockStaking.transferStake(
            CONTRACT_COLDKEY,
            CONTRACT_HOTKEY,
            NETUID,
            NETUID,
            ALPHA_AMOUNT
        );

        uint256 stakeAfterTransfer = mockStaking.getStake(
            CONTRACT_HOTKEY,
            CONTRACT_COLDKEY,
            NETUID
        );

        assertEq(stakeAfterTransfer, initialStake + ALPHA_AMOUNT);
    }
}
