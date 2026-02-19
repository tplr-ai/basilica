// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.24;

import {Script, console} from "forge-std/Script.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import {CollateralUpgradeable} from "../src/CollateralUpgradeable.sol";

contract CollateralUpgradeableUpgradeMock is CollateralUpgradeable {
    function getVersion() external pure override returns (uint256) {
        return 2;
    }

    function initializeUpgradeMock() external reinitializer(2) {}
}

contract DeployUpgradeableScript is Script {
    CollateralUpgradeable public collateral;
    ERC1967Proxy public proxy;

    function setUp() public {}

    function run() public {
        // Get deployment parameters from environment or use defaults
        uint16 netuid = uint16(vm.envOr("NETUID", uint256(39)));
        console.log("sender address is :", msg.sender);
        console.log("msg.sender balance is :", msg.sender.balance);
        address trustee = vm.envOr("TRUSTEE_ADDRESS", msg.sender);
        uint256 minCollateralIncrease = vm.envOr("MIN_COLLATERAL", uint256(1 ether));
        uint64 decisionTimeout = uint64(vm.envOr("DECISION_TIMEOUT", uint256(3600))); // 1 hour
        address admin = vm.envOr("ADMIN_ADDRESS", msg.sender);
        bytes32 validatorHotkey = vm.envOr("VALIDATOR_HOTKEY", bytes32(uint256(uint160(msg.sender))));

        console.log("Deploying Upgradeable Collateral contract with:");
        console.log("- NETUID:", netuid);
        console.log("- Trustee:", trustee);
        console.log("- Min Collateral:", minCollateralIncrease);
        console.log("- Decision Timeout:", decisionTimeout);
        console.log("- Admin:", admin);
        console.log("- Validator Hotkey:");
        console.logBytes32(validatorHotkey);

        vm.startBroadcast();

        // Deploy the implementation contract
        CollateralUpgradeable implementation = new CollateralUpgradeable();
        console.log("Implementation deployed at:", address(implementation));

        // Prepare initialization data
        bytes memory initData = abi.encodeWithSelector(
            CollateralUpgradeable.initialize.selector,
            netuid,
            trustee,
            minCollateralIncrease,
            decisionTimeout,
            admin,
            validatorHotkey
        );

        // Deploy the proxy with initialization
        proxy = new ERC1967Proxy(address(implementation), initData);
        console.log("Proxy deployed at:", address(proxy));

        // Cast proxy to interface for interaction
        collateral = CollateralUpgradeable(payable(address(proxy)));

        vm.stopBroadcast();

        console.log("Deployment completed!");
        console.log("Proxy Address (use this for interactions):", address(proxy));
        console.log("Implementation Address:", address(implementation));

        // Verify deployment
        console.log("\nVerification:");
        console.log("- NETUID:", collateral.netuid());
        console.log("- TRUSTEE:", collateral.trustee());
        console.log("- MIN_COLLATERAL_INCREASE:", collateral.minCollateralIncrease());
        console.log("- DECISION_TIMEOUT:", collateral.decisionTimeout());
        console.log("- VERSION:", collateral.getVersion());
    }

    /// @notice Deploy upgrade-mock implementation for upgrade testing
    function deployUpgradeMockImplementation() public {
        console.log("Deploying upgrade mock implementation...");

        vm.startBroadcast();

        CollateralUpgradeableUpgradeMock implementationV2 = new CollateralUpgradeableUpgradeMock();
        console.log("Upgrade mock implementation deployed at:", address(implementationV2));

        vm.stopBroadcast();

        console.log("Upgrade mock deployment completed!");
        console.log("Use this address for upgrading existing proxy for tests");
    }

    /// @notice Backward-compatible alias.
    function deployV2Implementation() public {
        deployUpgradeMockImplementation();
    }
}
