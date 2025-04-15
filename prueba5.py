label_encoder = LabelEncoder()
for col in df.select_dtypes(include = ["object"]).columns:
  df[col] = label_encoder.fit_transform(df[col])