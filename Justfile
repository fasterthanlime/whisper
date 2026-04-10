list:
    just --list

gen:
    cargo run --bin bee-rpc-codegen
    cd swift && xcodegen generate --spec bee-project.yml

install:
    scripts/install-bee.sh

install-debug:
    BEE_CONFIGURATION=debug scripts/install-bee.sh

verbose:
    scripts/debug-bee.sh

debug:
    scripts/debug-bee.sh --lldb
