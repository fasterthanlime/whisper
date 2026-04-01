# beeml-corrupt

Phoneme-based ASR corruption engine. Given a technical term, finds plausible
ASR confusions (single-word and multi-word) using phoneme edit distance.

## Setup

Requires two data files (not checked into git):

### CMUdict

```bash
curl -sL "https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b" -o data/cmudict.txt
```

### Phonetisaurus G2P model

Pre-trained FST for grapheme-to-phoneme conversion on OOV words.

```bash
# Download and extract
curl -sL "https://github.com/AdolfVonKleist/phonetisaurus-downloads/raw/master/g014b2b.tgz" -o /tmp/g014b2b.tgz
tar xzf /tmp/g014b2b.tgz -C /tmp/

# Compile the FST (requires openfst: brew install openfst)
cd /tmp/g014b2b && bash compile-fst.sh

# Copy to project
cp /tmp/g014b2b/g014b2b.fst models/g2p.fst
```
