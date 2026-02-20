import pickle

path1 = "../data/CSL-Daily/gloss2ids.pkl"
path2 = "../data/phoenix2014-T/gloss2ids.pkl"

with open(path1, "rb") as f:
    loaded_object = pickle.load(f)

with open(path2, "rb") as f:
    loaded_object2 = pickle.load(f)

common_keys = []
common_vals = []
for key in loaded_object.keys():
    if key in loaded_object2 and "<" in key:
        common_keys.append(key)
        common_vals.append(loaded_object[key])

print(common_keys)
print(common_vals)
combined_gloss2ids = {common_keys[i]: common_vals[i] for i in range(len(common_keys))}


c = len(combined_gloss2ids)
for key, value in loaded_object.items():
    if key not in combined_gloss2ids:
        combined_gloss2ids[key] = c
        c += 1

for key, value in loaded_object2.items():
    if key not in combined_gloss2ids:
        combined_gloss2ids[key] = c
        c += 1

# save
with open("../data/combined/gloss2ids.pkl", "wb") as f:
    pickle.dump(combined_gloss2ids, f)

# check if any keys have same values
value_to_keys = {}
for key, value in combined_gloss2ids.items():
    if value not in value_to_keys:
        value_to_keys[value] = []
    value_to_keys[value].append(key)

duplicate_values = [key for key, values in value_to_keys.items() if len(values) > 1]
print("Duplicate values:", duplicate_values)
print("Duplicate keys for duplicate values:", {value: value_to_keys[value] for value in duplicate_values})


print(len(common_keys))
print(len(loaded_object))
print(len(loaded_object2))
print(len(combined_gloss2ids))