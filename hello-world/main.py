from pathlib import Path

def main():
    print(f"Computing sum of integers 1 to 10")
    total = sum(range(1, 11))
    print(f"Sum = {total}")

    result_path = Path("/artifacts/result.txt")
    result_path.write_text(f"sum(1..10) = {total}\n")
    print(f"Result written to {result_path} (pull results with `job pull`)")


if __name__ == "__main__":
    main()
