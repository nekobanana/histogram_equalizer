import re

def get_results(results_file):
    execution_times = {}
    with open(results_file, "r") as f:
        lines = f.readlines()
    image_w = None
    image_h = None
    for line in lines:
        match = re.search("^Size (\d+)x(\d+)$", line)
        if match:
            image_w = match.groups()[0]
            image_h = match.groups()[1]
            continue
        match = re.search("^Average time:\s*(\d+.?\d*)$", line)
        if match:
            if image_w is None or image_h is None:
                continue
            run_time = match.groups()[0]
            execution_times[(f"{image_w}x{image_h}", int(image_w)*int(image_h))] = run_time
    return execution_times

def write_csv(input_file, output_file):
    with open(output_file, "w") as f:
        f.seek(0)
        f.write("image_size; image_size_pixels; execution_time\n")
        for (image_size, image_size_pixels), time in get_results(input_file).items():
            f.write(f"{image_size}; {image_size_pixels}; {time}\n")

if __name__ == "__main__":
    write_csv("../seq/results/results.txt", "results_seq.csv")
    write_csv("../parallel/results/results_16x16.txt", "results_par_16x16.csv")
    write_csv("../parallel/results/results_32x16.txt", "results_par_32x16.csv")
    write_csv("../parallel/results/results_20x16.txt", "results_par_20x16.csv")
    write_csv("../parallel/results/results_32x32.txt", "results_par_32x32.csv")
    write_csv("../parallel/results/results_32x24.txt", "results_par_32x24.csv")
