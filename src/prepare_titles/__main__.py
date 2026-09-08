from base_agent.config import load_config

from .dataset import filter_titles_by_buyers, save_titles_csv


def resolve_out_csv(out_csv: str, min_unique_buyers: int) -> str:
    """Fill the optional ``{min_unique_buyers}`` placeholder so the file name follows the threshold."""
    return out_csv.format(min_unique_buyers=min_unique_buyers)


def main() -> None:
    config = load_config()
    records = filter_titles_by_buyers(
        purchases_csv=config["purchases_csv"],
        min_unique_buyers=config["min_unique_buyers"],
    )
    save_titles_csv(
        records=records,
        out_path=resolve_out_csv(config["out_csv"], config["min_unique_buyers"]),
    )


if __name__ == "__main__":
    main()
