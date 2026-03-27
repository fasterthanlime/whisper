# Release Process for peft-rs

## Publication Information

**Crates.io**: Published as stable release (1.0.0)
**Maintainer**: Tyler Zervas (tzervas)  
**Contact**: tz-dev@vectorweight.com  
**GPG Key**: Required for all releases

## Pre-Release Checklist

### Code Quality
- [ ] All clippy warnings resolved (`cargo clippy -- -D warnings`)
- [ ] All tests passing (`cargo test --all-features`)
- [ ] Code formatted (`cargo fmt --check`)
- [ ] No security vulnerabilities (`cargo audit`)
- [ ] Documentation complete (`cargo doc --no-deps`)

### Version Management
- [ ] Version bumped in `Cargo.toml`
- [ ] `CHANGELOG.md` updated with all changes
- [ ] Git tag created: `vX.Y.Z`
- [ ] All changes committed to appropriate branch

### Branch Workflow
- [ ] Changes merged to `dev`
- [ ] Integrated and tested in `testing` branch
- [ ] Final validation in `testing`
- [ ] Merged to `main` for release

## GPG Signing Configuration

### Required GPG Key Details
```
Name: Tyler Zervas
Email: tz-dev@vectorweight.com
Username: tzervas
```

### Setup GPG Signing

1. **Verify GPG key exists**:
   ```bash
   gpg --list-secret-keys tz-dev@vectorweight.com
   ```

2. **Configure Git to use GPG**:
   ```bash
   git config --global user.signingkey <KEY_ID>
   git config --global commit.gpgsign true
   git config --global tag.gpgsign true
   ```

3. **Verify signing works**:
   ```bash
   echo "test" | gpg --clearsign
   ```

### Creating Signed Tags

```bash
# Create signed release tag
git tag -s v1.0.0 -m "Release v1.0.0: Stable Release"

# Verify tag signature
git tag -v v1.0.0

# Push tag to remote
git push origin v1.0.0
```

## Publishing to Crates.io

### Authentication
Already configured via `cargo login` with API token.

### Pre-Publication Verification

```bash
# Dry run (doesn't actually publish)
cargo publish --dry-run

# Check what will be included
cargo package --list

# Verify the package
cargo package
tar -tzf target/package/peft-rs-1.0.0.crate
```

### Publication Commands

#### Publishing Releases

```bash
# Publish to crates.io
cargo publish

# Verify publication
cargo search peft-rs
```

#### Version Naming Convention
- **1.0.x**: Stable API with all core PEFT adapters
- **1.x.y**: Backward-compatible feature additions and bug fixes
- **2.0.0+**: Future major versions with potential breaking changes

### Post-Publication

1. **Verify on crates.io**:
   - Visit: https://crates.io/crates/peft-rs
   - Check documentation builds
   - Verify metadata

2. **Create GitHub Release**:
   ```bash
   # Using GitHub CLI
   gh release create v1.0.0 \
     --title "v1.0.0 - Stable Release" \
     --notes-file RELEASE_NOTES.md
   ```

3. **Update Documentation**:
   - Link to crates.io in README.md
   - Update installation instructions
   - Add badges (version, downloads, docs)

## Release Types

### Stable Release (1.0.0+)
- API stability guarantee
- Full documentation
- Production validated
- All core PEFT adapters implemented

**Current Status**: 1.0.0 = **Stable release**

### Patch Releases (1.0.x)
- Bug fixes only
- No API changes
- Backward compatible

### Minor Releases (1.x.0)
- New features
- Backward compatible
- API additions only

## Emergency Procedures

### Yanking a Release
```bash
# If critical bug found
cargo yank --vers 1.0.0

# To un-yank (if fixed quickly)
cargo yank --vers 1.0.0 --undo
```

### Publishing Patches
```bash
# For critical fixes
# Version: 1.0.0 -> 1.0.1
cargo publish
```

## Current Publication Status

### Version 1.0.0
**Status**: Published as stable release

**Checklist**:
- All clippy warnings fixed
- 128 tests passing
- Comprehensive documentation
- All core PEFT adapters implemented
- Published to crates.io

## Automation Considerations

### Future CI/CD Integration
```yaml
# .github/workflows/release.yml (future)
on:
  push:
    tags:
      - 'v*'

jobs:
  publish:
    - verify GPG signature
    - run quality checks
    - cargo publish (with API token secret)
    - create GitHub release
```

## Version History

| Version | Date | Type | Published | Notes |
|---------|------|------|-----------|-------|
| 0.1.0 | 2026-01-06 | Dev | No | Initial adapters |
| 0.2.0 | 2026-01-07 | Dev | No | Additional adapters |
| 0.3.0 | 2026-01-08 | Dev | No | Infrastructure complete |
| 0.4.0 | 2026-01-09 | Alpha | No | Quality & tooling |
| 1.0.0 | 2026-01-24 | Stable | Yes | Stable release |

## References

- [Cargo Book - Publishing](https://doc.rust-lang.org/cargo/reference/publishing.html)
- [Crates.io](https://crates.io)
- [Semantic Versioning](https://semver.org)
- [GPG Signing Guide](https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work)

---

**Note**: This is a stable 1.0.0 release with API stability guarantees.
