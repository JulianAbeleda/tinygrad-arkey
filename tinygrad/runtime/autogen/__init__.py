import glob, importlib, os, pathlib, shutil, subprocess, tarfile, tempfile
from tinygrad.helpers import fetch, flatten, system, getenv

root = (here:=pathlib.Path(__file__).parent).parents[2]
ffmpeg_src = "https://ffmpeg.org/releases/ffmpeg-8.0.1.tar.gz"
rocr_src = "https://github.com/ROCm/rocm-systems/archive/refs/tags/rocm-7.1.1.tar.gz"
linux_headers_deb = "https://snapshot.debian.org/archive/debian/20260207T145350Z/pool/main/l/linux/linux-libc-dev_6.18.9-1_all.deb"
linux_headers_kern_deb = "https://snapshot.debian.org/archive/debian/20260207T145350Z/pool/main/l/linux/linux-headers-6.18.9+deb14-common_6.18.9-1_all.deb"
liburing_src = "https://raw.githubusercontent.com/axboe/liburing/refs/tags/liburing-2.14/src/include/liburing.h"
ggml_common_src = "https://raw.githubusercontent.com/ggml-org/ggml/d4fcfe88a8bcf5c9840be14be6c2fbf1f5b3b2db/src/ggml-common.h"
cudart_src = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.0.146-archive.tar.xz"
nvrtc_src = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-12.0.140-archive.tar.xz"
opencl_src = "https://github.com/KhronosGroup/OpenCL-Headers/archive/2e30669d48718fd460f085b4b35b160dad51ce9d.tar.gz"
macossdk = "/var/db/xcode_select_link/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"

llvm_lib = (
  (win_llvm:=r"'C:\\Program Files\\LLVM\\bin\\LLVM-C.dll' if WIN else ") +
  (mac_llvm:=repr([f'/opt/homebrew/opt/llvm@{i}/lib/libLLVM.dylib' for i in reversed(range(14, 21+1))]) + " if OSX else ") +
  (other_llvm:=repr(['LLVM'] + [f'LLVM-{i}' for i in reversed(range(14, 21+1))])))
clang_lib = win_llvm.replace("LLVM-C", "libclang") + (mac_llvm + other_llvm).replace("LLVM", "clang")

nv_lib_path = ("[f'/{pre}/cuda/targets/{tgt}/lib' for pre in ['opt', 'usr/local'] for tgt in "
               "[sysconfig.get_config_vars().get(\"MULTIARCH\", \"\").rsplit(\"-\", 1)[0], 'sbsa-linux']]")

def load(name, files, **kwargs):
  if not (f:=(root/(path:=kwargs.pop("path", __name__)).replace('.','/')/f"{name}.py")).exists() or getenv('REGEN'):
    files, kwargs['args'] = files() if callable(files) else files, args() if callable(args:=kwargs.get('args', [])) else args
    if (srcs:=kwargs.pop('srcs', None)):
      srcpath = (td:=tempfile.TemporaryDirectory(f"autogen-src-{name.replace('/','-')}")).name + "/"
      for src in (srcs if isinstance(srcs, list) else [srcs]):
        if 'tar' in src:
          # dangerous for arbitrary urls!
          with tarfile.open(fetch(src, gunzip=src.endswith("gz"))) as tf:
            tf.extractall(srcpath)
            if not isinstance(srcs, list): srcpath += tf.getnames()[0] # if we just have a single tarball, make this the root
        else: fetch(src, name=srcpath + src.split('/')[-1])
      files, kwargs['args'] = [str(f).format(srcpath) for f in files], [a.format(srcpath) for a in kwargs.get('args', [])]
      kwargs['anon_names'] = {k.format(srcpath):v for k,v in kwargs.get('anon_names', {}).items()}
      if (preprocess:=kwargs.pop('preprocess', None)): preprocess(srcpath)
    files = flatten(sorted(glob.glob(p, recursive=True)) if isinstance(p, str) and '*' in p else [p] for p in files)
    kwargs['epilog'] = (epi(srcpath) if srcs else epi()) if callable(epi:=kwargs.get('epilog', [])) else epi
    try: f.write_text(kwargs.pop("gen", importlib.import_module("tinygrad.runtime.support.autogen").gen)(name, files, **kwargs))
    except Exception as e: raise RuntimeError(f"error while generating {name}") from e
    if srcs: td.cleanup()
  return importlib.import_module(f"{path}.{name.replace('/', '.')}")

def __getattr__(nm):
  match nm:
    case "libc":
      return load("libc", lambda: ([i for i in system("dpkg -L libc6-dev").split() if 'sys/mman.h' in i or 'bits/mman-shared.h' in i] +
                                   ["/usr/include/string.h", "/usr/include/elf.h", "/usr/include/unistd.h", "/usr/include/asm-generic/mman-common.h"]),
                  args=["-D__USE_GNU", "-D_GNU_SOURCE"], dll="'c'", errno=True, recsym=True)
    case "avcodec": return load("avcodec", ["{}/libavcodec/hevc/hevc.h", "{}/libavcodec/cbs_h265.h"], srcs=ffmpeg_src)
    case "opencl": return load("opencl", ["{}/CL/cl.h"], dll="'OpenCL'", args=["-I{}"], srcs=opencl_src)
    case "cuda": return load("cuda", ["{}/include/cuda.h"], dll="'nvcuda' if WIN else 'cuda'", args=["-D__CUDA_API_VERSION_INTERNAL"], srcs=cudart_src, macros=False, prolog=["from tinygrad.helpers import WIN"])
    case "nvrtc": return load("nvrtc", ["{}/include/nvrtc.h"], dll="'nvrtc'", paths=nv_lib_path, srcs=nvrtc_src, prolog=["import sysconfig"])
    case "nvjitlink": load("nvjitlink", [root/"extra/nvJitLink.h"], dll="'nvJitLink'", paths=nv_lib_path, prolog=["import sysconfig"])
    case "kfd": return load("kfd", [root/"extra/hip_gpu_driver/kfd_ioctl.h"])
    # this defines all syscall numbers. should probably unify linux autogen?
    case "io_uring":
      return load("io_uring", ["{}/liburing.h", "{}/usr/include/linux/io_uring.h", "{}/usr/include/asm-generic/unistd.h"],
                  args=["-I{}/usr/include"], srcs=[linux_headers_deb, liburing_src], rules=[('__NR', 'NR')],
                  preprocess=lambda path: subprocess.run(f"ar x {linux_headers_deb.split('/')[-1]} && tar xf data.tar.xz", cwd=path, shell=True, check=True))
    case "llvm": return load("llvm", lambda: [system("llvm-config-20 --includedir")+"/llvm-c/**/*.h"], dll=llvm_lib,
                             args=lambda: system("llvm-config-20 --cflags").split(), recsym=True, prolog=["from tinygrad.helpers import WIN, OSX"])
    case "pci": return load("pci", ["{}/usr/include/linux/pci_regs.h"], srcs=linux_headers_deb,
                             preprocess=lambda path: subprocess.run(f"ar x {linux_headers_deb.split('/')[-1]} && tar xf data.tar.xz", cwd=path, shell=True, check=True))
    case "vfio": return load("vfio", ["{}/usr/include/linux/vfio.h"], args=["-I{}/usr/include"], srcs=linux_headers_deb,
                             preprocess=lambda path: subprocess.run(f"ar x {linux_headers_deb.split('/')[-1]} && tar xf data.tar.xz", cwd=path, shell=True, check=True))
    # could add rule: WGPU_COMMA -> ','
    case "libusb": return load("libusb", ["/usr/include/libusb-1.0/libusb.h"], dll="'usb-1.0'")
    case "hip": return load("hip", ["/opt/rocm/include/hip/hip_ext.h", "/opt/rocm/include/hip/hiprtc.h",
                                    "/opt/rocm/include/hip/hip_runtime_api.h", "/opt/rocm/include/hip/driver_types.h"],
                            dll="os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libamdhip64.so'",
                            args=["-D__HIP_PLATFORM_AMD__", "-I/opt/rocm/include", "-x", "c++"], prolog=["import os"])
    case "comgr" | "comgr_3":
      return load("comgr_3" if nm == "comgr_3" else "comgr", ["/opt/rocm/include/amd_comgr/amd_comgr.h"],
                  dll= "[os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libamd_comgr.so', 'amd_comgr']",
                  args=["-D__HIP_PLATFORM_AMD__", "-I/opt/rocm/include", "-x", "c++"], prolog=["import os"])
    case "hsa": return load("hsa", [*[f"{{}}/projects/rocr-runtime/runtime/hsa-runtime/core/inc/{s}.h" for s in ["registers"]],
                                    *[f"{{}}/projects/rocr-runtime/runtime/hsa-runtime/inc/{s}.h" for s in [
                                        "hsa", "hsa_ext_amd", "amd_hsa_signal", "amd_hsa_queue", "amd_hsa_kernel_code",
                                        "hsa_ext_finalize", "hsa_ext_image", "hsa_ven_amd_aqlprofile"]]],
      srcs=rocr_src, args=["-DLITTLEENDIAN_CPU"], prolog=["import os"])
    case "amdgpu_kd": return load("amdgpu_kd", lambda: [f"{system('llvm-config-20 --includedir')}/llvm/Support/AMDHSAKernelDescriptor.h"],
                                  args=lambda: system("llvm-config-20 --cflags").split() + ["-x", "c++"], recsym=True, macros=False)
    case "amd_gpu": return load("amd_gpu", [root/f"extra/hip_gpu_driver/{s}.h" for s in ["sdma_registers", "nvd", "gc_11_0_0_offset",
                                                                                               "sienna_cichlid_ip_offset"]],
                                args=["-I/opt/rocm/include", "-x", "c++"])
    case "amdgpu_drm": return load("amdgpu_drm", [ "/usr/include/drm/drm.h", *[root/f"extra/hip_gpu_driver/{s}.h" for s in ["amdgpu_drm"]]])
    case "kgsl": return load("kgsl", [root/"extra/qcom_gpu_driver/msm_kgsl.h"], args=["-D__user="])
    case "sqtt": return load("sqtt", [root/"extra/sqtt/sqtt.h"])
    case "rocprof":
      return load("rocprof", [f"{{}}/include/{s}.h" for s in ["rocprof_trace_decoder", "trace_decoder_instrument", "trace_decoder_types"]],
                  dll= "['rocprof-trace-decoder', p:='/usr/local/lib/rocprof-trace-decoder.so', p.replace('so','dylib')]",
                  srcs="https://github.com/ROCm/rocprof-trace-decoder/archive/dd0485100971522cc4cd8ae136bdda431061a04d.tar.gz")
    case "mesa": return load("mesa", [
        *[f"{{}}/src/compiler/nir/{s}.h" for s in ["nir", "nir_builder", "nir_shader_compiler_options", "nir_serialize"]], "{}/gen/nir_intrinsics.h",
        *[f"{{}}/src/nouveau/{s}.h" for s in ["headers/nv_device_info", "compiler/nak"]],
        *[f"{{}}/src/gallium/auxiliary/gallivm/lp_bld{s}.h" for s in ["", "_passmgr", "_misc", "_type", "_init", "_nir", "_struct", "_jit_types",
                                                                     "_flow", "_const"]],
        *[f"{{}}/src/freedreno/{s}.h" for s in ["common/freedreno_dev_info", "ir3/ir3_compiler", "ir3/ir3_shader", "ir3/ir3_nir"]],
        "{}/src/compiler/glsl_types.h", "{}/src/util/blob.h", "{}/src/util/ralloc.h", "{}/gen/ir3-isa.h", "{}/gen/builtin_types.h",
        "{}/gen/a6xx.xml.h", "{}/gen/adreno_pm4.xml.h", "{}/gen/a6xx_enums.xml.h", "{}/gen/a6xx_descriptors.xml.h"], args=lambda:[
          "-DHAVE_ENDIAN_H", "-DHAVE_STRUCT_TIMESPEC", "-DHAVE_PTHREAD", "-DHAVE_FUNC_ATTRIBUTE_PACKED", "-I{}/src", "-I{}/include", "-I{}/gen",
          "-I{}/src/compiler/nir", "-I{}/src/gallium/auxiliary", "-I{}/src/gallium/include", "-I{}/src/freedreno/common",
          f"-I{system('llvm-config-20 --includedir')}"],
        preprocess=lambda path: subprocess.run("\n".join(["mkdir -p gen/util/format", "python3 src/compiler/builtin_types_h.py gen/builtin_types.h",
          "python3 src/compiler/isaspec/decode.py --xml src/freedreno/isa/ir3.xml --out-c /dev/null --out-h gen/ir3-isa.h",
          "python3 src/util/format/u_format_table.py src/util/format/u_format.yaml --enums > gen/util/format/u_format_gen.h",
          *["python3 src/freedreno/registers/gen_header.py --rnn src/freedreno/registers/ --xml " +
            f"src/freedreno/registers/adreno/{s}.xml c-defines > gen/{s}.xml.h" for s in ["a6xx", "adreno_pm4", "a6xx_enums", "a6xx_descriptors"]],
          *[f"python3 src/compiler/{s}_h.py > gen/{s.split('/')[-1]}.h" for s in ["nir/nir_opcodes", "nir/nir_builder_opcodes"]],
          *[f"python3 src/compiler/nir/nir_{s}_h.py --outdir gen" for s in ["intrinsics", "intrinsics_indices"]]]), cwd=path, shell=True, check=True),
  srcs="https://gitlab.freedesktop.org/mesa/mesa/-/archive/mesa-25.2.7/mesa-25.2.7.tar.gz",
  dll="'tinymesa_cpu' if (_cpu:=DEV.renderer == 'LVP') else 'tinymesa', " \
      'emsg="not available on this platform" if WIN or (OSX and (platform.machine() != "arm64" or (_mv:=platform.mac_ver()[0][:2]) not in {"14","15","26"})) or (platform.system() == "Linux" and platform.machine() not in {"x86_64", "aarch64"}) else ' \
      'f"run `sudo curl -fL https://github.com/sirhcm/tinymesa/releases/download/v1/libtinymesa{\'_cpu\'*_cpu}-mesa-25.2.7-{\'macos-\'+_mv if OSX else \'linux\'}-{\'amd64\' if ARCH_X86 else \'arm64\'}.{\'dylib\' if OSX else \'so\'} -o /usr/local/lib/libtinymesa{\'_cpu\'*_cpu}.{\'dylib\' if OSX else \'so\'}`"',
  prolog=["from tinygrad.helpers import DEV, ARCH_X86, WIN, OSX", "import gzip, base64, platform"],
  epilog=lambda path: [system(f"{root}/extra/mesa/lvp_nir_options.sh {path}")])
    case "libclang":
      return load("libclang",
                  lambda: [f"{system('llvm-config-20 --includedir')}/clang-c/{s}.h" for s in ["Index", "CXString", "CXSourceLocation", "CXFile"]],
                  dll=clang_lib, prolog=["from tinygrad.helpers import WIN, OSX"], args=lambda: system("llvm-config-20 --cflags").split())
    case "metal":
      return load("metal", [f"{macossdk}/System/Library/Frameworks/Metal.framework/Headers/MTL{s}.h" for s in
                  ["ComputeCommandEncoder", "ComputePipeline", "CommandQueue", "Device", "IndirectCommandBuffer", "Resource", "CommandEncoder"]],
                  dll="'Metal'", args=["-xobjective-c","-isysroot",macossdk], types={"dispatch_data_t":"objc.id_"})
    case "iokit": return load("iokit", [f"{macossdk}/System/Library/Frameworks/IOKit.framework/Headers/IOKitLib.h"], dll="'IOKit'",
                              args=["-isysroot", macossdk])
    case "corefoundation": return load("corefoundation",
                                       [f"{macossdk}/System/Library/Frameworks/CoreFoundation.framework/Headers/CF{s}.h" for s in ["String", "Data"]],
                                       dll="'CoreFoundation'",args=["-isysroot", macossdk])
    case "llvm_qcom": return load("llvm_qcom", [root/"extra/tinydreno.h"], dll="'llvm-qcom'")
    case "ggml_common": return load("ggml_common", ["{}/ggml-common.h"], srcs=ggml_common_src,
                                    args=["-DGGML_COMMON_DECL_C", "-DGGML_COMMON_IMPL_C"], macros=False)
    case "mlx5":
      kh = "{}/usr/src/linux-headers-6.18.9+deb14-common/include/linux/mlx5"
      return load("mlx5", [root/"extra/mlx_driver/mlx5.h", f"{kh}/mlx5_ifc.h"], srcs=linux_headers_kern_deb,
                  args=["-Du8=unsigned char", "-Du16=unsigned short", "-Du32=unsigned int", "-Du64=unsigned long long",
                        "-D__be16=unsigned short", "-D__be32=unsigned int", "-D__be64=unsigned long long", f"-I{kh}"],
                  preprocess=lambda path: subprocess.run(f"ar x {linux_headers_kern_deb.split('/')[-1]} && tar xf data.tar.xz",
                                                         cwd=path, shell=True, check=True))
    case _: raise AttributeError(f"no such autogen: {nm}")
