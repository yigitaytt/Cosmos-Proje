from datasets import load_dataset

ds = load_dataset(
    "BILGEM-AI/BILGE-Synthetic-Math",
    split="train"
)

# Örneğin 40 parçaya böl (yaklaşık 100MB civarı olur)
shards_count = 300

shards = ds.shard(num_shards=shards_count, index=0)

# Bir tanesini kaydetmek için:
for i in range(shards_count):
    shard = ds.shard(num_shards=shards_count, index=i)
    shard.save_to_disk(f"/data/shard_{i}")

print('Dataset pull completed.')