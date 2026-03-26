#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "bittensor>=9.0.0",
#   "textual>=1.0.0",
#   "click>=8.1.0",
# ]
# [tool.uv]
# prerelease = "allow"
# ///

"""
BitTensor Subnet Viewer TUI

Interactive terminal UI for inspecting BitTensor subnet configuration,
hyperparameters, neuron metagraph, and emission data.

Usage:
    ./subnet-viewer.py                              # List all subnets (finney)
    ./subnet-viewer.py --netuid 1                   # View subnet 1 details
    ./subnet-viewer.py --network test --netuid 1    # Testnet subnet 1
    ./subnet-viewer.py --network ws://custom:9944   # Custom node
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass

import click

# Parse CLI args with click BEFORE importing bittensor, which hijacks argparse.
# We stash parsed values and proceed after.
_cli_network: str = "finney"
_cli_netuid: int | None = None


@click.command()
@click.option("--network", default="finney", help="Network: finney/test/local/custom URL")
@click.option("--netuid", default=None, type=int, help="Subnet UID (omit for list view)")
def _parse_cli(network: str, netuid: int | None) -> None:
    global _cli_network, _cli_netuid
    _cli_network = network
    _cli_netuid = netuid


_result = _parse_cli(standalone_mode=False)
if isinstance(_result, int):
    # --help was invoked (click returns 0), exit cleanly
    sys.exit(_result)

# Now safe to import bittensor (it will see an empty sys.argv)
_orig_argv = sys.argv
sys.argv = sys.argv[:1]

import bittensor  # noqa: E402

sys.argv = _orig_argv

logging.getLogger("bittensor").setLevel(logging.WARNING)

from textual import work  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import VerticalScroll  # noqa: E402
from textual.screen import Screen  # noqa: E402
from textual.widgets import (  # noqa: E402
    Collapsible,
    DataTable,
    Footer,
    Header,
    LoadingIndicator,
    Static,
    TabbedContent,
    TabPane,
)


def truncate_key(key: str) -> str:
    """Truncate SS58 key for display."""
    if not key or len(key) < 20:
        return key or "N/A"
    return f"{key[:8]}...{key[-8:]}"


def fmt_val(value, fallback: str = "N/A") -> str:
    """Format a value for display, handling None and Balance types."""
    if value is None:
        return fallback
    try:
        return str(value)
    except Exception:
        return fallback


FIELD_DESCRIPTIONS: dict[str, str] = {
    # --- Emissions ---
    "subnet_emission": "Legacy pre-dTAO field, hardcoded to 0 on chain. Superseded by tao_in/alpha_in/alpha_out emission",
    "alpha_in_emission": "Alpha tokens injected into the subnet's liquidity pool each block (tao_in / price)",
    "alpha_out_emission": "Alpha distributed to miners (41%), validators (41%), and subnet owner (18%)",
    "tao_in_emission": "TAO injected into the AMM pool each block; may be less than full allocation if alpha injection cap triggers",
    "pending_alpha_emission": "Accumulated alpha awaiting distribution at next Yuma Consensus epoch (~tempo blocks)",
    "pending_root_emission": "Accumulated TAO equivalents bound for root subnet (SN0) stakers",
    # --- Pool / Reserve ---
    "alpha_out": "Total alpha emitted and held outside the pool (in user hotkeys)",
    "alpha_in": "Alpha tokens held in the subnet's AMM liquidity reserve",
    "tao_in": "TAO tokens held in the subnet's AMM liquidity reserve",
    "subnet_volume": "Cumulative staking transaction volume in the subnet's pool",
    "moving_price": "EMA price of alpha (TAO_reserve / Alpha_reserve), smoothed ~30-day half-life",
    # --- Consensus / Weights ---
    "min_allowed_weights": "Minimum number of UIDs a validator must include in each weight-set",
    "max_weights_limit": "Max normalized weight a validator can assign to any single UID",
    "weights_version": "Required weights version; validators below this are excluded from consensus",
    "weights_rate_limit": "Minimum blocks between successive weight commits by one hotkey",
    "alpha_high": "Upper bound of the Liquid Alpha range for Yuma Consensus mixing",
    "alpha_low": "Lower bound of the Liquid Alpha range for Yuma Consensus mixing",
    "alpha_sigmoid_steepness": "Steepness of the sigmoid mapping consensus score to the alpha mixing factor",
    "liquid_alpha_enabled": "Whether per-UID adaptive alpha is used instead of a fixed value",
    "commit_reveal_weights_enabled": "Whether commit-reveal protocol is required for weight submissions",
    "commit_reveal_period": "Tempo intervals between weight commit and reveal phases",
    "rho": "Temperature parameter controlling weight normalization in Yuma Consensus",
    "kappa": "Shift/threshold parameter in the Yuma Consensus sigmoid function",
    # --- Registration ---
    "registration_allowed": "Whether new neuron registrations are open on this subnet",
    "pow_registration_allowed": "Whether proof-of-work registration is accepted (vs burn-only)",
    "immunity_period": "Blocks a newly registered neuron is protected from deregistration",
    "burn": "Current TAO burn cost to register a neuron on this subnet",
    "difficulty": "Current PoW difficulty target for proof-of-work registration",
    "min_difficulty": "Floor on PoW difficulty; cannot adjust below this",
    "max_difficulty": "Ceiling on PoW difficulty; cannot adjust above this",
    "min_burn": "Floor on burn cost; cannot adjust below this",
    "max_burn": "Ceiling on burn cost; cannot adjust above this",
    "adjustment_alpha": "EMA smoothing factor for registration rate adjustments",
    "adjustment_interval": "Blocks between difficulty/burn recalculations",
    "target_regs_per_interval": "Desired registrations per adjustment interval",
    "max_regs_per_block": "Max neuron registrations allowed in a single block",
    "serving_rate_limit": "Minimum blocks between successive axon-serve calls from one hotkey",
    # --- Neuron / Validator ---
    "num_uids": "Current number of active neurons registered on the subnet",
    "max_uids": "Maximum neuron slots available on the subnet",
    "max_validators": "Max neurons that can hold a validator permit",
    "activity_cutoff": "Blocks of inactivity after which a validator's weights are ignored",
    # --- Bonds ---
    "bonds_moving_avg": "EMA decay factor for bond accumulation between validators and miners",
    "bonds_reset_enabled": "Whether bonds are periodically reset to zero",
    # --- State ---
    "subnet_is_active": "Whether the subnet is active and processing consensus",
    "transfers_enabled": "Whether alpha token transfers between hotkeys are allowed",
    "user_liquidity_enabled": "Whether users can add/remove liquidity to the subnet's AMM pool",
    "yuma_version": "Yuma Consensus algorithm version running on this subnet",
}


# ---------------------------------------------------------------------------
# Subnet List Screen
# ---------------------------------------------------------------------------


class SubnetListScreen(Screen):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
    ]

    CSS = """
    #loading { height: auto; }
    .hidden { display: none; }
    #subnet-table { height: 1fr; }
    """

    def __init__(self, network: str) -> None:
        super().__init__()
        self.network = network

    def compose(self) -> ComposeResult:
        yield Header()
        yield LoadingIndicator(id="loading")
        yield DataTable(id="subnet-table", classes="hidden")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#subnet-table", DataTable)
        table.cursor_type = "row"
        table.add_columns(
            "NetUID", "Name", "Owner", "Tempo", "Neurons", "Max UIDs", "Emission"
        )
        self.load_subnets()

    @work(thread=True)
    def load_subnets(self) -> None:
        try:
            sub = bittensor.Subtensor(network=self.network)
            subnets = sub.get_all_subnets_info()
        except Exception as e:
            self.app.call_from_thread(self._show_error, str(e))
            return

        self.app.call_from_thread(self._populate_table, subnets)

    def _show_error(self, msg: str) -> None:
        self.query_one("#loading").add_class("hidden")
        table = self.query_one("#subnet-table", DataTable)
        table.remove_class("hidden")
        self.notify(f"Error: {msg}", severity="error", timeout=10)

    def _populate_table(self, subnets) -> None:
        self.query_one("#loading").add_class("hidden")
        table = self.query_one("#subnet-table", DataTable)
        table.remove_class("hidden")
        table.clear()

        for sn in subnets:
            netuid = getattr(sn, "netuid", "?")
            name = getattr(sn, "subnet_name", "") or getattr(sn, "name", "") or ""
            owner = truncate_key(getattr(sn, "owner_ss58", "") or "")
            tempo = fmt_val(getattr(sn, "tempo", None))
            neurons = fmt_val(getattr(sn, "subnetwork_n", None))
            max_uids = fmt_val(getattr(sn, "max_n", None))
            emission = fmt_val(getattr(sn, "emission_value", None))
            table.add_row(str(netuid), name, owner, tempo, neurons, max_uids, emission, key=str(netuid))

    def action_refresh(self) -> None:
        self.query_one("#loading").remove_class("hidden")
        self.query_one("#subnet-table", DataTable).add_class("hidden")
        self.load_subnets()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        netuid = int(event.row_key.value)
        self.app.push_screen(SubnetDetailScreen(self.network, netuid))

    def action_quit(self) -> None:
        self.app.exit()


# ---------------------------------------------------------------------------
# Subnet Detail Screen
# ---------------------------------------------------------------------------


@dataclass
class SubnetData:
    """Container for fetched subnet data."""
    network: str
    netuid: int
    metagraph: object
    hyperparams: object


class SubnetDetailScreen(Screen):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("b", "go_back", "Back"),
        Binding("r", "refresh", "Refresh"),
    ]

    CSS = """
    #loading { height: auto; }
    .hidden { display: none; }
    #detail-content { height: 1fr; }
    .section-label { color: $text-muted; margin: 1 0 0 0; }
    .field-row { margin: 0 0 0 2; }
    .field-name { color: $accent; min-width: 30; }
    .field-value { color: $text; }
    #metagraph-table { height: 1fr; }
    """

    def __init__(self, network: str, netuid: int) -> None:
        super().__init__()
        self.network = network
        self.netuid = netuid
        self.subnet_data: SubnetData | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield LoadingIndicator(id="loading")
        with TabbedContent(id="detail-content", classes="hidden"):
            with TabPane("Identity", id="tab-identity"):
                yield VerticalScroll(id="identity-content")
            with TabPane("Hyperparameters", id="tab-hparams"):
                yield VerticalScroll(id="hparams-content")
            with TabPane("Metagraph", id="tab-metagraph"):
                yield DataTable(id="metagraph-table")
            with TabPane("Emissions", id="tab-emissions"):
                yield VerticalScroll(id="emissions-content")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"Subnet {self.netuid}"
        table = self.query_one("#metagraph-table", DataTable)
        table.cursor_type = "row"
        self.load_data()

    @work(thread=True)
    def load_data(self) -> None:
        try:
            sub = bittensor.Subtensor(network=self.network)
            metagraph = sub.metagraph(netuid=self.netuid, lite=True)
            hyperparams = sub.get_subnet_hyperparameters(netuid=self.netuid)
        except Exception as e:
            self.app.call_from_thread(self._show_error, str(e))
            return

        data = SubnetData(network=self.network, netuid=self.netuid, metagraph=metagraph, hyperparams=hyperparams)
        self.app.call_from_thread(self._render_all, data)

    def _show_error(self, msg: str) -> None:
        self.query_one("#loading").add_class("hidden")
        self.query_one("#detail-content").remove_class("hidden")
        self.notify(f"Error: {msg}", severity="error", timeout=10)

    def _render_all(self, data: SubnetData) -> None:
        self.subnet_data = data
        self.query_one("#loading").add_class("hidden")
        self.query_one("#detail-content").remove_class("hidden")

        mg = data.metagraph
        name = getattr(mg, "name", None) or getattr(mg, "subnet_name", None) or f"Subnet {self.netuid}"
        self.sub_title = f"{name} (netuid {self.netuid})"

        self._render_identity(mg)
        self._render_hyperparams(mg, data.hyperparams)
        self._render_metagraph(mg)
        self._render_emissions(mg)

    # --- Identity Tab ---

    def _render_identity(self, mg) -> None:
        container = self.query_one("#identity-content")
        container.remove_children()

        identity = getattr(mg, "identity", None)  # dict or None

        fields = [
            ("Subnet Name", getattr(mg, "name", None)),
            ("Symbol", getattr(mg, "symbol", None)),
            ("Owner Coldkey", getattr(mg, "owner_coldkey", None)),
            ("Owner Hotkey", getattr(mg, "owner_hotkey", None)),
            ("Network Registered At", getattr(mg, "network_registered_at", None)),
        ]

        if identity and isinstance(identity, dict):
            fields.extend([
                ("Description", identity.get("description")),
                ("GitHub Repo", identity.get("github_repo")),
                ("Subnet URL", identity.get("subnet_url")),
                ("Discord", identity.get("discord")),
                ("Logo URL", identity.get("logo_url")),
                ("Contact", identity.get("subnet_contact")),
                ("Additional", identity.get("additional")),
            ])
        else:
            fields.append(("Identity", "Not set on-chain"))

        for label, value in fields:
            display_val = fmt_val(value, "Not set")
            container.mount(Static(f"  [bold]{label}:[/bold]  {display_val}"))

    # --- Hyperparameters Tab ---

    def _render_hyperparams(self, mg, hp) -> None:
        container = self.query_one("#hparams-content")
        container.remove_children()

        hparams = getattr(mg, "hparams", None)

        def _get(obj, attr, fallback="N/A"):
            if obj is None:
                return fallback
            val = getattr(obj, attr, None)
            if val is None:
                return fallback
            return str(val)

        # hparams = mg.hparams (human-readable values)
        # hp = SubnetHyperparameters (supplemental fields marked with *)
        sections = {
            "Consensus / Weights": [
                ("min_allowed_weights", hp),
                ("max_weights_limit", hparams),
                ("weights_version", hp),
                ("weights_rate_limit", hp),
                ("alpha_high", hparams),
                ("alpha_low", hparams),
                ("alpha_sigmoid_steepness", hp),
                ("liquid_alpha_enabled", hparams),
                ("commit_reveal_weights_enabled", hparams),
                ("commit_reveal_period", hp),
                ("rho", hparams),
                ("kappa", hparams),
            ],
            "Registration": [
                ("registration_allowed", hparams),
                ("pow_registration_allowed", hparams),
                ("immunity_period", hp),
                ("burn", hparams),
                ("difficulty", hparams),
                ("min_difficulty", hparams),
                ("max_difficulty", hparams),
                ("min_burn", hparams),
                ("max_burn", hparams),
                ("adjustment_alpha", hparams),
                ("adjustment_interval", hp),
                ("target_regs_per_interval", hp),
                ("max_regs_per_block", hp),
                ("serving_rate_limit", hp),
            ],
            "Neuron / Validator": [
                ("num_uids", mg),
                ("max_uids", mg),
                ("max_validators", hp),
                ("activity_cutoff", hp),
            ],
            "Bonds": [
                ("bonds_moving_avg", hparams),
                ("bonds_reset_enabled", hp),
            ],
            "State": [
                ("subnet_is_active", hp),
                ("transfers_enabled", hp),
                ("user_liquidity_enabled", hp),
                ("yuma_version", hp),
            ],
        }

        for section_name, params in sections.items():
            lines = []
            for attr, source in params:
                val = _get(source, attr)
                lines.append(f"  [bold]{attr}:[/bold]  {val}")
                desc = FIELD_DESCRIPTIONS.get(attr, "")
                if desc:
                    lines.append(f"    [dim italic]{desc}[/dim italic]")

            content = "\n".join(lines)
            container.mount(
                Collapsible(
                    Static(content),
                    title=section_name,
                    collapsed=False,
                )
            )

    # --- Metagraph Tab ---

    # TODO: Show alpha tokens given out to each miner in metagraph over last 24hr / 7day.
    #       Requires querying historical blocks (using archive node) and summing
    #       alpha_out_emission distributed to each miner UID across those windows.

    def _render_metagraph(self, mg) -> None:
        table = self.query_one("#metagraph-table", DataTable)
        table.clear(columns=True)
        table.add_columns(
            "UID", "Hotkey", "Coldkey", "Total Stake", "Alpha Stake",
            "TAO Stake", "Trust", "Consensus", "Incentive", "Dividends",
            "Emission", "Active", "Val.Permit", "Axon",
        )

        n = getattr(mg, "n", 0) or 0
        if n == 0:
            n = len(getattr(mg, "uids", []))

        rows = []
        axons = getattr(mg, "axons", None)
        for i in range(n):
            stake_raw = self._neuron_float(mg, "total_stake", i)

            axon_ip = ""
            if axons is not None and i < len(axons):
                axon = axons[i]
                ip = getattr(axon, "ip", "") or ""
                port = getattr(axon, "port", "") or ""
                if ip and ip != "0.0.0.0":
                    axon_ip = f"{ip}:{port}"

            rows.append((stake_raw, (
                self._neuron_val(mg, "uids", i, str(i)),
                truncate_key(self._neuron_val(mg, "hotkeys", i, "")),
                truncate_key(self._neuron_val(mg, "coldkeys", i, "")),
                self._neuron_num(mg, "total_stake", i),
                self._neuron_num(mg, "alpha_stake", i),
                self._neuron_num(mg, "tao_stake", i),
                self._neuron_num(mg, "trust", i),
                self._neuron_num(mg, "consensus", i),
                self._neuron_num(mg, "incentive", i),
                self._neuron_num(mg, "dividends", i),
                self._neuron_num(mg, "emission", i),
                str(self._neuron_val(mg, "active", i, "?")),
                str(self._neuron_val(mg, "validator_permit", i, "?")),
                axon_ip,
            )))

        rows.sort(key=lambda r: r[0], reverse=True)
        for _, cols in rows:
            table.add_row(*cols, key=cols[0])

    def _neuron_val(self, mg, attr: str, idx: int, fallback: str = "") -> str:
        arr = getattr(mg, attr, None)
        if arr is None:
            return fallback
        try:
            val = arr[idx]
            if hasattr(val, "item"):
                return str(val.item())
            return str(val)
        except (IndexError, TypeError):
            return fallback

    def _neuron_float(self, mg, attr: str, idx: int) -> float:
        arr = getattr(mg, attr, None)
        if arr is None:
            return 0.0
        try:
            val = arr[idx]
            if hasattr(val, "item"):
                val = val.item()
            return float(val)
        except (IndexError, TypeError, ValueError):
            return 0.0

    def _neuron_num(self, mg, attr: str, idx: int) -> str:
        arr = getattr(mg, attr, None)
        if arr is None:
            return "N/A"
        try:
            val = arr[idx]
            if hasattr(val, "item"):
                val = val.item()
            fval = float(val)
            if fval == int(fval) and abs(fval) < 1e12:
                return str(int(fval))
            return f"{fval:.6f}"
        except (IndexError, TypeError, ValueError):
            return "N/A"

    # --- Emissions Tab ---

    def _render_emissions(self, mg) -> None:
        container = self.query_one("#emissions-content")
        container.remove_children()

        emissions = getattr(mg, "emissions", None)
        pool = getattr(mg, "pool", None)

        emission_fields = [
            ("subnet_emission", emissions),
            ("alpha_in_emission", emissions),
            ("alpha_out_emission", emissions),
            ("tao_in_emission", emissions),
            ("pending_alpha_emission", emissions),
            ("pending_root_emission", emissions),
        ]

        pool_fields = [
            ("alpha_out", pool),
            ("alpha_in", pool),
            ("tao_in", pool),
            ("subnet_volume", pool),
            ("moving_price", pool),
        ]

        container.mount(Static("  [bold underline]Emissions[/bold underline]"))
        for field, source in emission_fields:
            val = getattr(source, field, None) if source else None
            desc = FIELD_DESCRIPTIONS.get(field, "")
            desc_line = f"\n      [dim italic]{desc}[/dim italic]" if desc else ""
            container.mount(Static(f"    [bold]{field}:[/bold]  {fmt_val(val)}{desc_line}"))

        container.mount(Static(""))
        container.mount(Static("  [bold underline]Pool / Reserve[/bold underline]"))
        for field, source in pool_fields:
            val = getattr(source, field, None) if source else None
            desc = FIELD_DESCRIPTIONS.get(field, "")
            desc_line = f"\n      [dim italic]{desc}[/dim italic]" if desc else ""
            container.mount(Static(f"    [bold]{field}:[/bold]  {fmt_val(val)}{desc_line}"))

    # --- Actions ---

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "metagraph-table":
            return
        if self.subnet_data is None:
            return
        uid = int(event.row_key.value)
        self.app.push_screen(ValidatorWeightsScreen(
            network=self.subnet_data.network,
            netuid=self.subnet_data.netuid,
            validator_uid=uid,
            subnet_data=self.subnet_data,
        ))

    def action_refresh(self) -> None:
        self.query_one("#loading").remove_class("hidden")
        self.query_one("#detail-content").add_class("hidden")
        self.load_data()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_quit(self) -> None:
        self.app.exit()


# ---------------------------------------------------------------------------
# Validator Weights Helper
# ---------------------------------------------------------------------------


def extract_validator_weights(all_weights, validator_uid: int) -> list[tuple[int, float]]:
    """Extract and sort a single validator's weights from subtensor.weights() result."""
    for uid, weights in all_weights:
        if uid == validator_uid:
            result = [(tuid, float(w)) for tuid, w in weights]
            return sorted(result, key=lambda x: x[1], reverse=True)
    return []


# ---------------------------------------------------------------------------
# Validator Weights Screen
# ---------------------------------------------------------------------------


class ValidatorWeightsScreen(Screen):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("b", "go_back", "Back"),
        Binding("c", "compare", "Compare"),
        Binding("r", "refresh", "Refresh"),
    ]

    CSS = """
    #loading { height: auto; }
    .hidden { display: none; }
    #weights-content { height: 1fr; }
    #weights-table { height: 1fr; }
    #history-table { height: 1fr; }
    #validator-info { padding: 1 2; }
    """

    def __init__(self, network: str, netuid: int, validator_uid: int, subnet_data: SubnetData) -> None:
        super().__init__()
        self.network = network
        self.netuid = netuid
        self.validator_uid = validator_uid
        self.subnet_data = subnet_data
        self.all_weights: list | None = None
        self.current_block: int | None = None
        self.tempo: int | None = None
        self._history_loaded = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield LoadingIndicator(id="loading")
        with TabbedContent(id="weights-content", classes="hidden"):
            with TabPane("Current Weights", id="tab-current"):
                yield Static(id="validator-info")
                yield DataTable(id="weights-table")
            with TabPane("History", id="tab-history"):
                yield Static("Switch to this tab to load history...", id="history-status")
                yield DataTable(id="history-table")
        yield Footer()

    def on_mount(self) -> None:
        mg = self.subnet_data.metagraph
        uid = self.validator_uid
        hotkey = self._neuron_val(mg, "hotkeys", uid, "N/A")
        stake = self._neuron_num(mg, "total_stake", uid)
        self.sub_title = f"Validator UID {uid} Weights"

        info = self.query_one("#validator-info", Static)
        info.update(
            f"[bold]UID:[/bold] {uid}  |  "
            f"[bold]Hotkey:[/bold] {truncate_key(hotkey)}  |  "
            f"[bold]Stake:[/bold] {stake}"
        )

        table = self.query_one("#weights-table", DataTable)
        table.cursor_type = "row"

        hist_table = self.query_one("#history-table", DataTable)
        hist_table.cursor_type = "row"

        self.load_weights()

    def _neuron_val(self, mg, attr: str, idx: int, fallback: str = "") -> str:
        arr = getattr(mg, attr, None)
        if arr is None:
            return fallback
        try:
            val = arr[idx]
            if hasattr(val, "item"):
                return str(val.item())
            return str(val)
        except (IndexError, TypeError):
            return fallback

    def _neuron_num(self, mg, attr: str, idx: int) -> str:
        arr = getattr(mg, attr, None)
        if arr is None:
            return "N/A"
        try:
            val = arr[idx]
            if hasattr(val, "item"):
                val = val.item()
            fval = float(val)
            if fval == int(fval) and abs(fval) < 1e12:
                return str(int(fval))
            return f"{fval:.6f}"
        except (IndexError, TypeError, ValueError):
            return "N/A"

    @work(thread=True)
    def load_weights(self) -> None:
        try:
            sub = bittensor.Subtensor(network=self.network)
            self.all_weights = sub.weights(netuid=self.netuid)
            self.current_block = sub.block
            hp = self.subnet_data.hyperparams
            self.tempo = getattr(hp, "tempo", None)
            if self.tempo is not None:
                self.tempo = int(self.tempo)
        except Exception as e:
            self.app.call_from_thread(self._show_error, str(e))
            return

        self.app.call_from_thread(self._render_weights)

    def _show_error(self, msg: str) -> None:
        self.query_one("#loading").add_class("hidden")
        self.query_one("#weights-content").remove_class("hidden")
        self.notify(f"Error: {msg}", severity="error", timeout=10)

    def _render_weights(self) -> None:
        self.query_one("#loading").add_class("hidden")
        self.query_one("#weights-content").remove_class("hidden")

        weights = extract_validator_weights(self.all_weights, self.validator_uid)

        table = self.query_one("#weights-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Miner UID", "Hotkey", "Weight", "% of Total")

        total = sum(w for _, w in weights)
        mg = self.subnet_data.metagraph
        for miner_uid, weight in weights:
            pct = (weight / total * 100) if total > 0 else 0.0
            hotkey = self._neuron_val(mg, "hotkeys", miner_uid, "N/A")
            table.add_row(
                str(miner_uid),
                truncate_key(hotkey),
                f"{weight:.6f}",
                f"{pct:.2f}%",
            )

        if not weights:
            self.notify("No weights set by this validator", severity="warning")

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.pane.id == "tab-history" and not self._history_loaded:
            self._history_loaded = True
            self._load_history()

    @work(thread=True)
    def _load_history(self) -> None:
        if self.all_weights is None or self.current_block is None or self.tempo is None:
            self.app.call_from_thread(
                self._update_history_status, "Cannot load history: missing block/tempo info"
            )
            return

        current = extract_validator_weights(self.all_weights, self.validator_uid)
        top_uids = [uid for uid, _ in current[:20]]

        if not top_uids:
            self.app.call_from_thread(
                self._update_history_status, "No current weights — cannot show history"
            )
            return

        self.app.call_from_thread(self._setup_history_table, top_uids)

        archive_network = "archive" if self.network in ("finney", "archive") else self.network
        sub = bittensor.Subtensor(network=archive_network)
        loaded = 0
        errors = 0
        first_error = None
        for i in range(1, 11):
            block = self.current_block - self.tempo * i
            if block <= 0:
                break
            try:
                hist_weights = sub.weights(netuid=self.netuid, block=block)
                validator_w = extract_validator_weights(hist_weights, self.validator_uid)
                w_dict = dict(validator_w)
                age_blocks = self.current_block - block
                self.app.call_from_thread(
                    self._add_history_row, block, age_blocks, top_uids, w_dict
                )
                loaded += 1
            except Exception as e:
                errors += 1
                if first_error is None:
                    first_error = str(e)
                continue

        if loaded == 0 and first_error:
            msg = f"No historical data available ({errors} queries failed: {first_error})"
        elif loaded == 0:
            msg = "No historical data available"
        else:
            msg = f"History loaded ({loaded} epochs{f', {errors} failed' if errors else ''})"
        self.app.call_from_thread(self._update_history_status, msg)

    def _update_history_status(self, msg: str) -> None:
        self.query_one("#history-status", Static).update(msg)

    def _setup_history_table(self, top_uids: list[int]) -> None:
        table = self.query_one("#history-table", DataTable)
        table.clear(columns=True)
        cols = ["Block", "Age"] + [f"UID {uid}" for uid in top_uids]
        table.add_columns(*cols)
        self._update_history_status("Loading history...")

    def _add_history_row(self, block: int, age_blocks: int, top_uids: list[int], w_dict: dict) -> None:
        table = self.query_one("#history-table", DataTable)
        row = [str(block), f"{age_blocks} blocks"]
        for uid in top_uids:
            w = w_dict.get(uid, 0.0)
            row.append(f"{w:.6f}" if w > 0 else "—")
        table.add_row(*row)

    def action_compare(self) -> None:
        if self.all_weights is None:
            self.notify("Weights not loaded yet", severity="warning")
            return
        self.app.push_screen(ValidatorPickerScreen(
            network=self.network,
            netuid=self.netuid,
            validator_a_uid=self.validator_uid,
            all_weights=self.all_weights,
            subnet_data=self.subnet_data,
        ))

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_refresh(self) -> None:
        self._history_loaded = False
        self.query_one("#loading").remove_class("hidden")
        self.query_one("#weights-content").add_class("hidden")
        self.load_weights()

    def action_quit(self) -> None:
        self.app.exit()


# ---------------------------------------------------------------------------
# Validator Picker Screen
# ---------------------------------------------------------------------------


class ValidatorPickerScreen(Screen):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("b", "go_back", "Back"),
    ]

    CSS = """
    #picker-table { height: 1fr; }
    """

    def __init__(
        self,
        network: str,
        netuid: int,
        validator_a_uid: int,
        all_weights: list,
        subnet_data: SubnetData,
    ) -> None:
        super().__init__()
        self.network = network
        self.netuid = netuid
        self.validator_a_uid = validator_a_uid
        self.all_weights = all_weights
        self.subnet_data = subnet_data

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="picker-table")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "Pick validator to compare"

        table = self.query_one("#picker-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("UID", "Hotkey", "Stake")

        mg = self.subnet_data.metagraph
        n = getattr(mg, "n", 0) or 0
        if n == 0:
            n = len(getattr(mg, "uids", []))

        validators = []
        permits = getattr(mg, "validator_permit", None)
        for i in range(n):
            if permits is not None:
                try:
                    perm = permits[i]
                    if hasattr(perm, "item"):
                        perm = perm.item()
                    if not perm:
                        continue
                except (IndexError, TypeError):
                    continue

            if i == self.validator_a_uid:
                continue

            stake_raw = 0.0
            stake_arr = getattr(mg, "total_stake", None)
            if stake_arr is not None:
                try:
                    val = stake_arr[i]
                    if hasattr(val, "item"):
                        val = val.item()
                    stake_raw = float(val)
                except (IndexError, TypeError, ValueError):
                    pass

            uid_val = i
            uids_arr = getattr(mg, "uids", None)
            if uids_arr is not None:
                try:
                    uid_val = uids_arr[i]
                    if hasattr(uid_val, "item"):
                        uid_val = uid_val.item()
                except (IndexError, TypeError):
                    pass

            hotkey = ""
            hk_arr = getattr(mg, "hotkeys", None)
            if hk_arr is not None:
                try:
                    hotkey = str(hk_arr[i])
                except (IndexError, TypeError):
                    pass

            validators.append((stake_raw, str(uid_val), hotkey))

        validators.sort(key=lambda v: v[0], reverse=True)
        for stake_raw, uid_str, hotkey in validators:
            stake_display = f"{stake_raw:.6f}" if stake_raw != int(stake_raw) else str(int(stake_raw))
            table.add_row(uid_str, truncate_key(hotkey), stake_display, key=uid_str)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        validator_b_uid = int(event.row_key.value)
        self.app.push_screen(ValidatorCompareScreen(
            network=self.network,
            netuid=self.netuid,
            validator_a_uid=self.validator_a_uid,
            validator_b_uid=validator_b_uid,
            all_weights=self.all_weights,
            subnet_data=self.subnet_data,
        ))

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_quit(self) -> None:
        self.app.exit()


# ---------------------------------------------------------------------------
# Validator Compare Screen
# ---------------------------------------------------------------------------


class ValidatorCompareScreen(Screen):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("b", "go_back", "Back"),
    ]

    CSS = """
    #compare-table { height: 1fr; }
    #compare-summary { padding: 1 2; }
    """

    def __init__(
        self,
        network: str,
        netuid: int,
        validator_a_uid: int,
        validator_b_uid: int,
        all_weights: list,
        subnet_data: SubnetData,
    ) -> None:
        super().__init__()
        self.network = network
        self.netuid = netuid
        self.validator_a_uid = validator_a_uid
        self.validator_b_uid = validator_b_uid
        self.all_weights = all_weights
        self.subnet_data = subnet_data

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield DataTable(id="compare-table")
            yield Static(id="compare-summary")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"Compare UID {self.validator_a_uid} vs UID {self.validator_b_uid}"

        weights_a = extract_validator_weights(self.all_weights, self.validator_a_uid)
        weights_b = extract_validator_weights(self.all_weights, self.validator_b_uid)

        dict_a = dict(weights_a)
        dict_b = dict(weights_b)
        all_miner_uids = sorted(set(dict_a.keys()) | set(dict_b.keys()))

        table = self.query_one("#compare-table", DataTable)
        table.cursor_type = "row"
        table.add_columns(
            "Miner UID", "Hotkey",
            f"Val {self.validator_a_uid} Weight", f"Val {self.validator_b_uid} Weight",
            "Diff",
        )

        rows = []
        for muid in all_miner_uids:
            wa = dict_a.get(muid, 0.0)
            wb = dict_b.get(muid, 0.0)
            diff = abs(wa - wb)
            rows.append((diff, muid, wa, wb))

        rows.sort(key=lambda r: r[0], reverse=True)

        mg = self.subnet_data.metagraph
        for diff, muid, wa, wb in rows:
            hotkey = ""
            hk_arr = getattr(mg, "hotkeys", None)
            if hk_arr is not None:
                try:
                    hotkey = str(hk_arr[muid])
                except (IndexError, TypeError):
                    pass
            table.add_row(
                str(muid),
                truncate_key(hotkey),
                f"{wa:.6f}" if wa > 0 else "—",
                f"{wb:.6f}" if wb > 0 else "—",
                f"{diff:.6f}",
            )

        # Summary
        set_a = set(dict_a.keys())
        set_b = set(dict_b.keys())
        overlap = len(set_a & set_b)
        only_a = len(set_a - set_b)
        only_b = len(set_b - set_a)
        cos_sim = self._cosine_similarity(dict_a, dict_b)

        summary = self.query_one("#compare-summary", Static)
        summary.update(
            f"[bold]Overlap:[/bold] {overlap} miners weighted by both  |  "
            f"[bold]Only Val {self.validator_a_uid}:[/bold] {only_a}  |  "
            f"[bold]Only Val {self.validator_b_uid}:[/bold] {only_b}  |  "
            f"[bold]Cosine Similarity:[/bold] {cos_sim:.10f}"
        )

    @staticmethod
    def _cosine_similarity(dict_a: dict, dict_b: dict) -> float:
        all_keys = set(dict_a.keys()) | set(dict_b.keys())
        dot = sum(dict_a.get(k, 0.0) * dict_b.get(k, 0.0) for k in all_keys)
        norm_a = math.sqrt(sum(v ** 2 for v in dict_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in dict_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_quit(self) -> None:
        self.app.exit()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class SubnetViewerApp(App):
    TITLE = "BitTensor Subnet Viewer"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, network: str = "finney", initial_netuid: int | None = None) -> None:
        super().__init__()
        self.network = network
        self.initial_netuid = initial_netuid

    def on_mount(self) -> None:
        if self.initial_netuid is not None:
            self.push_screen(SubnetDetailScreen(self.network, self.initial_netuid))
        else:
            self.push_screen(SubnetListScreen(self.network))


def main() -> None:
    app = SubnetViewerApp(network=_cli_network, initial_netuid=_cli_netuid)
    app.run()


if __name__ == "__main__":
    main()
