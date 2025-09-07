from pathlib import Path
import yaml

from marketing_simulation.website import Page, WebsiteGraph

def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text()) if p.exists() else {}

# src/marketing_simulation/config_utils.py
from typing import Dict, Tuple, List, Optional
from marketing_simulation.website import Page

def website_factory_from_yaml(cfg: dict) -> Dict[str, Page]:
    """
    Build Page objects from YAML.
    Backward compatible:
      - old keys:  account_creation: bool, conversion: bool
      - new keys:  account_creation_element: str, conversion_element: str
    """
    pages: Dict[str, Page] = {}

    # First pass: construct pages
    for name, spec in (cfg or {}).items():
        drop = float(spec.get("dropoff_prob", 0.0))
        elements = list(spec.get("clickable_elements", []) or [])

        # --- Backward compatibility mapping ---
        # Prefer explicit *element keys*; otherwise map bools to defaults.
        acct_el = spec.get("account_creation_element")
        if acct_el is None and bool(spec.get("account_creation", False)):
            acct_el = "account_creation"

        conv_el = spec.get("conversion_element")
        if conv_el is None and bool(spec.get("conversion", False)):
            conv_el = "submit_order"

        # Ensure elements list contains any special elements we rely on
        if acct_el and acct_el not in elements:
            elements.append(acct_el)
        if conv_el and conv_el not in elements:
            elements.append(conv_el)

        # Normalize transitions -> List[Tuple[str, float]]
        transitions_in = spec.get("transitions") or []
        transitions: List[Tuple[str, float]] = []
        for t in transitions_in:
            if isinstance(t, (list, tuple)) and len(t) == 2:
                nxt, prob = t[0], float(t[1])
                transitions.append((str(nxt), prob))

        pages[name] = Page(
            name=name,
            dropoff_prob=drop,
            clickable_elements=elements or None,
            account_creation_element=acct_el,
            conversion_element=conv_el,
            transitions=transitions or None,
        )

    # Optional: light validation (doesn't raise; logs by returning warnings list if you want)
    # Example: ensure transition targets exist
    missing_targets = []
    for pname, p in pages.items():
        if not p.transitions:
            continue
        for tgt, _ in p.transitions:
            if tgt not in pages:
                missing_targets.append((pname, tgt))
    if missing_targets:
        # Replace this with your logger if available
        print(f"[website_factory_from_yaml] WARNING: missing transition targets: {missing_targets}")

    return pages

def build_channel_plan(raw_cfg: dict | None) -> dict[str, dict]:
    """
    Normalize raw YAML channel config into the format run_simulation expects.

    Expected YAML:
      Organic:
        visitors: 120
        mean_gap_s: 20

    Returns dict like:
      {"Organic": {"visitors": 120, "mean_gap_s": 20.0}, ...}
    Falls back to defaults if no config provided.
    """
    if not raw_cfg:
        return {
            "Organic": {"visitors": 120, "mean_gap_s": 20.0},
            "Paid":    {"visitors":  80, "mean_gap_s": 15.0},
            "Referral":{"visitors":  40, "mean_gap_s": 30.0},
        }

    plan: dict[str, dict] = {}
    for ch, spec in raw_cfg.items():
        plan[ch] = {
            "visitors": int(spec.get("visitors", 0)),
            "mean_gap_s": float(spec.get("mean_gap_s", 0.0)),
        }
    return plan