// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "vox-unix-test",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../../../vox/swift/vox-runtime"),
    ],
    targets: [
        .executableTarget(
            name: "vox-unix-test",
            dependencies: [
                .product(name: "VoxRuntime", package: "vox-runtime"),
            ]
        ),
    ]
)
