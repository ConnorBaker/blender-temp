final: prev: {
  cudaPackages = final.cudaPackages_13_1;

  blender-render = final.callPackage ./blender-render { };

  blender-temp = final.python3.pkgs.callPackage ./default.nix { };

  openimagedenoise = prev.openimagedenoise.overrideAttrs (
    finalAttrs: prevAttrs: {
      version = "2.4.1";
      src = final.fetchzip {
        url = "https://github.com/RenderKit/oidn/releases/download/v${finalAttrs.version}/oidn-${finalAttrs.version}.src.tar.gz";
        hash = "sha256-SM0Bn4qgeqRJAXr2MMjNjfWJVTcciERZxMHiyx4Z1hA=";
      };
      patches = [ ./0001-oidn-cuda-fix.patch ];
    }
  );

  opensubdiv = prev.opensubdiv.overrideAttrs (prevAttrs: {
    patches = prevAttrs.patches or [ ] ++ [
      # Hopefully included after 3.7.0
      # https://github.com/PixarAnimationStudios/OpenSubdiv/pull/1378
      (final.fetchpatch2 {
        name = "fixed-deprecated-cuda-api.patch";
        url = "https://github.com/PixarAnimationStudios/OpenSubdiv/commit/cb1b2378c8fb370b4cc9b71079473145fed6ae35.patch";
        hash = "sha256-uqr7O3JOLdrDWsLRAgTZJGbTV8XQU+VgYvup+oHizOU=";
      })
    ];
  });

  _cuda = prev._cuda.extend (
    finalCuda: prevCuda: {
      extensions = prevCuda.extensions ++ [
        (finalCudaPackages: prevCudaPackages: {
          cuda_cccl = prevCudaPackages.cuda_cccl.overrideAttrs (prevAttrs: {
            # NVIDIA, in their wisdom, expect CCCL to be a directory inside include.
            # https://github.com/NVIDIA/cutlass/blob/087c84df83d254b5fb295a7a408f1a1d554085cf/CMakeLists.txt#L773
            postInstall = prevAttrs.postInstall or "" + ''
              nixLog "creating alias for ''${!outputInclude:?}/include/cccl"
              ln -srv "''${!outputInclude:?}/include" "''${!outputInclude:?}/include/cccl"
            '';
          });
        })
      ];
    }
  );

  pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
    (finalPythonPackages: prevPythonPackages: {
      helion = prevPythonPackages.helion.overridePythonAttrs (prevAttrs: {
        dependencies = prevAttrs.dependencies ++ [
          finalPythonPackages.scikit-learn
        ];

        patches = prevAttrs.patches or [] ++ [
          ./0001-Fix-None-indexing-to-append-implicit-trailing-slices.patch
          ./0002-Fix-unsqueeze-codegen-to-resolve-negative-dims-corre.patch
          ./0003-Deduplicate-tensor-indexer-dims-across-type-propagat.patch
        ];
      });

      torch = prevPythonPackages.torch.overrideAttrs (prevAttrs: {
        meta = prevAttrs.meta // {
          # CUDA 13 isn't allowed with PyTorch 2.10, see:
          # https://github.com/NixOS/nixpkgs/blob/nixos-unstable/pkgs/development/python-modules/torch/source/default.nix#L243
          broken = false;
        };
      });

      warp-lang =
        (prevPythonPackages.warp-lang.override {
          # Patch doesn't apply cleanly; it's fine, we only need the CUDA backend.
          # https://github.com/NixOS/nixpkgs/blob/eac9adc9cc293c4cec9686f9ae534cf21a5f7c7e/pkgs/development/python-modules/warp-lang/default.nix#L24
          standaloneSupport = false;
        }).overrideAttrs
          (prevAttrs: {
            buildInputs = prevAttrs.buildInputs ++ [
              final.cudaPackages.libnvptxcompiler
            ];
          });
    })
  ];
}
