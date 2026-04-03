use bee_phonetic::SeedDataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = SeedDataset::load_canonical()?;
    dataset.validate()?;
    let aliases = dataset.lexicon_aliases();
    let index = dataset.phonetic_index();

    println!("seed_root={}", dataset.root.display());
    println!("terms={}", dataset.terms.len());
    println!("sentence_examples={}", dataset.sentence_examples.len());
    println!("recording_examples={}", dataset.recording_examples.len());
    println!("aliases={}", aliases.len());
    println!("postings={}", index.postings.len());
    println!("phone_len_buckets={}", index.by_phone_len.len());
    println!("token_count_buckets={}", index.by_token_count.len());

    Ok(())
}
