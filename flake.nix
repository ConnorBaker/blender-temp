{
  inputs = {
    flake-parts = {
      inputs.nixpkgs-lib.follows = "nixpkgs";
      url = "github:hercules-ci/flake-parts";
    };
    nixpkgs.url = "github:nixos/nixpkgs";
    git-hooks-nix = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:cachix/git-hooks.nix";
    };
    treefmt-nix = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:numtide/treefmt-nix";
    };
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "aarch64-linux"
        "x86_64-linux"
      ];

      imports = [
        inputs.treefmt-nix.flakeModule
        inputs.git-hooks-nix.flakeModule
      ];

      flake.overlays.default = import ./overlay.nix;

      perSystem =
        {
          config,
          pkgs,
          system,
          ...
        }:
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
              cudaCapabilities = [ "8.9" ];
            };
            overlays = [ inputs.self.overlays.default ];
          };

          legacyPackages = pkgs;

          packages = {
            inherit (pkgs)
              blender-render
              blender-temp
              ;

            default = config.packages.blender-temp;
          };

          pre-commit.settings.hooks = {
            # Formatter checks
            treefmt = {
              enable = true;
              package = config.treefmt.build.wrapper;
            };

            # Nix checks
            deadnix.enable = true;
            nil.enable = true;
            statix.enable = true;

            # Python
            ruff.enable = true;
          };

          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              # Nix
              nixfmt.enable = true;

              # Python
              ruff-format.enable = true;

              # Shell
              shellcheck.enable = true;
              shfmt.enable = true;
            };
          };
        };
    };
}
