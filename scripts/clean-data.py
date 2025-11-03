from pyarrow import parquet

businesses = (
    parquet.read_table("data/compressed/yelp_academic_dataset_business.gz")
    .drop_columns("hours")
    .flatten()
)
photos = parquet.read_table("data/compressed/photos.gz")

df = photos.join(businesses, "business_id").combine_chunks()

print(df.schema)
print(df.shape)
print(df.take([1, 5]))
parquet.write_table(df, "data/clean/yelp.gz", compression="gzip")
