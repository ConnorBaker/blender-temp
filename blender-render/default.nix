{
  blender,
  lib,
  writeShellApplication,
}:
writeShellApplication {
  name = "blender-temp";
  runtimeInputs = [ blender ];
  text = ''
    exec blender --background -noaudio --python-exit-code 1 --python ${./render.py} -- "$@"
  '';
  meta = {
    description = "Render Blender scenes with configurable settings and multi-camera jitter";
    maintainers = with lib.maintainers; [ connorbaker ];
    mainProgram = "blender-temp";
  };
}
