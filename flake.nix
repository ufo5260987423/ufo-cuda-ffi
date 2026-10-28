{
  description = "Chez Scheme FFI bindings for the CUDA driver API (ufo-cuda)";

  inputs = {
    # Pinned to the nixpkgs already present in this machine's Nix store
    # (the system registry's flake:nixpkgs), so no network fetch of the
    # nixpkgs tree is needed.  To use this flake elsewhere, point the
    # input at your own nixpkgs, e.g.:
    #   nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs.url = "path:/nix/store/zc9xvdvz642n6q20m04ixm30s98m6281-source";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        # CUDA packages are unfree; this predicate allows exactly the CUDA set.
        config.allowUnfreePredicate = pkgs._cuda.lib.allowUnfreeCudaPredicate;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "ufo-cuda";
        packages = with pkgs; [
          chez
          cudaPackages.cudatoolkit
        ];

        shellHook = ''
          # Point the test suite at the driver stub when no GPU driver is
          # installed (this is a NixOS machine without an NVIDIA GPU).
          export CUDA_STUBS_DIR=${pkgs.cudaPackages.cudatoolkit}/lib/stubs
          export UFO_CUDA_LIB=${pkgs.cudaPackages.cudatoolkit}/lib/stubs/libcuda.so
          echo "ufo-cuda dev shell"
          echo "  nvcc:        $(command -v nvcc)"
          echo "  stub libcuda: $UFO_CUDA_LIB"
        '';
      };
    };
}
