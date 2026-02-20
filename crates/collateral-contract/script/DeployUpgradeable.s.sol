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
        // Required deployment parameters; missing env vars should fail fast.
        uint256 netuidRaw = vm.envUint("NETUID");
        uint256 decisionTimeoutRaw = vm.envUint("DECISION_TIMEOUT");
        require(netuidRaw <= type(uint16).max, "NETUID exceeds uint16 max");
        require(decisionTimeoutRaw <= type(uint64).max, "DECISION_TIMEOUT exceeds uint64 max");
        // casting is safe because bounds are enforced above
        // forge-lint: disable-next-line(unsafe-typecast)
        uint16 netuid = uint16(netuidRaw);
        // casting is safe because bounds are enforced above
        // forge-lint: disable-next-line(unsafe-typecast)
        uint64 decisionTimeout = uint64(decisionTimeoutRaw);

        address trustee = vm.envAddress("TRUSTEE_ADDRESS");
        uint256 minCollateralIncrease = vm.envUint("MIN_COLLATERAL");
        address admin = vm.envAddress("ADMIN_ADDRESS");
        bytes32 validatorHotkey = vm.envBytes32("VALIDATOR_HOTKEY");

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
