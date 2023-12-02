import { pipeline } from "@xenova/transformers";

async function main() {
  const pipe = await pipeline("feature-extraction", "Supabase/gte-small");

  // Generate the embedding from text
  const output = await pipe("Hello world", {
    pooling: "mean",
    normalize: true,
  });

  // Extract the embedding output
  const embedding = Array.from(output.data);

  console.log(JSON.stringify(embedding));
}

main();
