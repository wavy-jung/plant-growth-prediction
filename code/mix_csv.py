import pandas as pd

bc_path = "regnet_032_bc.csv"
lt_path = "regnet_032_lt.csv"
save_name = "bc_lt_regnetx32_fold20.csv"

bc_file = pd.read_csv(bc_path)
lt_file = pd.read_csv(lt_path)
bc_delta = bc_file['time_delta']
lt_delta = lt_file['time_delta']
test_file = pd.read_csv("../data/test_dataset/test_data.csv")
standard_list = [path.split("_")[1].lower() for path in list(test_file['before_file_path'])]

mix_time_delta = []
for idx in range(len(bc_file)):
    if standard_list[idx] == "bc":
        mix_time_delta.append(bc_delta[idx])
    elif standard_list[idx] == "lt":
        mix_time_delta.append(lt_delta[idx])

assert len(standard_list)==len(mix_time_delta), "Something Wrong"

df = pd.DataFrame({
    "idx": list(range(len(standard_list))),
    "time_delta": mix_time_delta
})

df.to_csv(save_name, index=False)
