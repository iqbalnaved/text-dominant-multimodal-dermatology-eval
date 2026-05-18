import os

def count_vignette_words(parent_dir):
    results = []

    # Loop through each subdirectory in the parent
    for subdir in os.listdir(parent_dir):
        subdir_path = os.path.join(parent_dir, subdir)

        if os.path.isdir(subdir_path):
            # Find the .txt file inside the subdir
            txt_files = [f for f in os.listdir(subdir_path) if f.endswith(".txt")]
            if not txt_files:
                continue  # skip if no text file

            file_name = txt_files[0]
            file_path = os.path.join(subdir_path, file_name)

            # Read the file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Split at the first occurrence of "Question"
            vignette = content.split("Question", 1)[0]

            # Count words in the vignette
            word_count = len(vignette.split())

            results.append((subdir, file_name, word_count))

    # Print output in the desired format
    print("subdir_name, file_name, word_count")
    for r in results:
        print(f"{r[0]}, {r[1]}, {r[2]}")

    return results

def summarize_counts(results):
    # Sort results by word count
    sorted_results = sorted(results, key=lambda x: x[2])

    print("\n=== Vignettes (Shorest to Longest) ===")
    for r in sorted_results:
        # print(f"{r[0]}, {r[1]}, {r[2]} words")
        print(f"{r[0]}, {r[2]}")
    # Bottom 10
    # bottom_10 = sorted_results[:10]

    # Top 10
    # top_10 = sorted_results[-10:]

    # print("\n=== Bottom 10 Subdirs (Shortest Vignettes) ===")
    # for r in bottom_10:
        # print(f"{r[0]}, {r[1]}, {r[2]} words")

    # print("\n=== Top 10 Subdirs (Longest Vignettes) ===")
    # for r in top_10:
        # print(f"{r[0]}, {r[1]}, {r[2]} words")


# Example usage:
# parent_directory = "/path/to/your/parent_dir"
# results = count_vignette_words(parent_directory)
# summarize_counts(results)


parent_directory = "/mnt/d/Naved/Data/jdcr_derm_vignette/2022-2025"
results = count_vignette_words(parent_directory)
summarize_counts(results)
