// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import {Script, console} from "forge-std/Script.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import {CollateralUpgradeable} from "../src/CollateralUpgradeable.sol";
import {CollateralUpgradeableV2} from "../src/CollateralUpgradeableV2.sol";

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
        uint256 minCollateralIncrease = vm.envOr(
            "MIN_COLLATERAL",
            uint256(1 ether)
        );
        uint64 decisionTimeout = uint64(
            vm.envOr("DECISION_TIMEOUT", uint256(3600))
        ); // 1 hour
        address admin = vm.envOr("ADMIN_ADDRESS", msg.sender);
        bytes32 validatorHotkey = vm.envOr(
            "VALIDATOR_HOTKEY",
            bytes32(uint256(uint160(msg.sender)))
        );

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
        console.log(
            "Proxy Address (use this for interactions):",
            address(proxy)
        );
        console.log("Implementation Address:", address(implementation));

        // Verify deployment
        console.log("\nVerification:");
        console.log("- NETUID:", collateral.NETUID());
        console.log("- TRUSTEE:", collateral.TRUSTEE());
        console.log(
            "- MIN_COLLATERAL_INCREASE:",
            collateral.MIN_COLLATERAL_INCREASE()
        );
        console.log("- DECISION_TIMEOUT:", collateral.DECISION_TIMEOUT());
        console.log("- VERSION:", collateral.getVersion());
    }

    /// @notice Deploy V2 implementation for upgrade testing
    function deployV2Implementation() public {
        console.log("Deploying V2 implementation...");

        vm.startBroadcast();

        CollateralUpgradeableV2 implementationV2 = new CollateralUpgradeableV2();
        console.log(
            "V2 Implementation deployed at:",
            address(implementationV2)
        );

        vm.stopBroadcast();

        console.log("V2 deployment completed!");
        console.log("Use this address for upgrading existing proxy to V2");
    }
}
