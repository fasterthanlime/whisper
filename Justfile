list:
    just --list

# Refresh SourceKit metadata for accurate Xcode diagnostics.
sourcekit-refresh:
    # sourcekit-lsp -> BSP -> xcode-build-server
    # xcode-build-server needs fresh Xcode logs to regenerate `.compile`
    # without `.compile`, SourceKit may use partial args and show false
    # "Cannot find type ... in scope" diagnostics.
    eval "$(direnv export bash)"
    cd swift && xcodebuild -project bee.xcodeproj -scheme bee -configuration Debug build
    xcode-build-server config -project swift/bee.xcodeproj -scheme bee
    build_root="$(awk -F\" '/"build_root"/ { print $4 }' buildServer.json)"; \
      xcode-build-server parse --sync "$build_root"

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

ime-spy:
    cd ime-spy && xcodegen generate --spec project.yml
    xcodebuild -project ime-spy/ime-spy.xcodeproj -scheme ime-spy -configuration Debug build 2>&1 | xcbeautify
    pkill ime-spy || true
    sleep 0.5
    @build_dir="$(xcodebuild -project ime-spy/ime-spy.xcodeproj -scheme ime-spy -configuration Debug -showBuildSettings 2>/dev/null | awk '/^ *BUILT_PRODUCTS_DIR =/ { print $3 }')"; \
      rm -rf /Applications/ime-spy.app; \
      cp -R "$build_dir/ime-spy.app" /Applications/ime-spy.app; \
      echo "Installed to /Applications/ime-spy.app"
    open /Applications/ime-spy.app
