{
  buildPythonPackage,
  cudaPackages,
  helion,
  hypothesis,
  lib,
  pythonOlder,
  pytest,
  warp-lang,
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
    warp-lang
    helion
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
