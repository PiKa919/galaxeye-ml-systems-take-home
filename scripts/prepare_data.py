from collections import Counter

from galaxeye.data import prepare_data

if __name__ == "__main__":
    result = prepare_data()
    print(f"Split digest: {result['digest']}")
    print(dict(Counter(record["split"] for record in result["records"])))
