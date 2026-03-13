{
  buildPythonPackage,
  cudaPackages,
  helion,
  hypothesis,
  lib,
  pythonOlder,
  pytest,
  warp-lang,
  scikit-learn,
  setuptools,
  torch,
  torchvision,
}:
let
  inherit (lib.fileset) toSource unions;
  inherit (lib.trivial) importTOML;
  pyprojectAttrs = importTOML ./pyproject.toml;
in
buildPythonPackage (finalAttrs: {
  pname = pyprojectAttrs.project.name;
  inherit (pyprojectAttrs.project) version;
  pyproject = true;
  disabled = pythonOlder "3.11";
  src = toSource {
    root = ./.;
    fileset = unions [
      ./pyproject.toml
      ./blender_temp
    ];
  };
  build-system = [ setuptools ];
  dependencies = [
    (warp-lang.override {
      # Patch doesn't apply cleanly; it's fine, we only need the CUDA backend.
      # https://github.com/NixOS/nixpkgs/blob/eac9adc9cc293c4cec9686f9ae534cf21a5f7c7e/pkgs/development/python-modules/warp-lang/default.nix#L24
      standaloneSupport = false;
    })
    (helion.overridePythonAttrs (prevAttrs: {
      dependencies = prevAttrs.dependencies ++ [
        scikit-learn
      ];
    }))
    hypothesis
    pytest
    torch
    torchvision
    # For Triton
    cudaPackages.backendStdenv.cc
    cudaPackages.cuda_nvcc
  ];
  pythonImportsCheck = [ finalAttrs.pname ];
  doCheck = true;
  meta = with lib; {
    inherit (pyprojectAttrs.project) description;
    maintainers = with maintainers; [ connorbaker ];
    mainProgram = "blender-temp";
  };
})
