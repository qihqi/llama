module @IrToHlo.6869 attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<32000x4096xf32>, %arg1: tensor<4096xf32>, %arg2: tensor<f32>, %arg3: tensor<4096x11008xf32>, %arg4: tensor<11008x4096xf32>, %arg5: tensor<4096xf32>, %arg6: tensor<4096x4096xf32>, %arg7: tensor<2049xi64>, %arg8: tensor<4096x4096xf32>, %arg9: tensor<4096xf32>, %arg10: tensor<4096x11008xf32>, %arg11: tensor<11008x4096xf32>, %arg12: tensor<4096xf32>, %arg13: tensor<4096x4096xf32>, %arg14: tensor<4096x4096xf32>, %arg15: tensor<4096xf32>, %arg16: tensor<4096x11008xf32>, %arg17: tensor<11008x4096xf32>, %arg18: tensor<4096xf32>, %arg19: tensor<4096x4096xf32>, %arg20: tensor<4096x4096xf32>, %arg21: tensor<4096xf32>, %arg22: tensor<4096x11008xf32>, %arg23: tensor<11008x4096xf32>, %arg24: tensor<4096xf32>, %arg25: tensor<4096x4096xf32>, %arg26: tensor<4096x4096xf32>, %arg27: tensor<4096xf32>, %arg28: tensor<4096x11008xf32>, %arg29: tensor<11008x4096xf32>, %arg30: tensor<4096xf32>, %arg31: tensor<4096x4096xf32>, %arg32: tensor<4096x4096xf32>, %arg33: tensor<4096xf32>, %arg34: tensor<4096x11008xf32>, %arg35: tensor<11008x4096xf32>, %arg36: tensor<4096xf32>, %arg37: tensor<4096x4096xf32>, %arg38: tensor<4096x4096xf32>, %arg39: tensor<4096xf32>, %arg40: tensor<4096x11008xf32>, %arg41: tensor<11008x4096xf32>, %arg42: tensor<4096xf32>, %arg43: tensor<4096x4096xf32>, %arg44: tensor<4096x4096xf32>, %arg45: tensor<4096xf32>, %arg46: tensor<4096x11008xf32>, %arg47: tensor<11008x4096xf32>, %arg48: tensor<4096xf32>, %arg49: tensor<4096x4096xf32>, %arg50: tensor<4096x4096xf32>, %arg51: tensor<4096xf32>, %arg52: tensor<4096x11008xf32>, %arg53: tensor<11008x4096xf32>, %arg54: tensor<4096xf32>, %arg55: tensor<4096x4096xf32>, %arg56: tensor<4096x4096xf32>, %arg57: tensor<4096xf32>, %arg58: tensor<4096x11008xf32>, %arg59: tensor<11008x4096xf32>, %arg60: tensor<4096xf32>, %arg61: tensor<4096x4096xf32>, %arg62: tensor<4096x4096xf32>, %arg63: tensor<4096xf32>, %arg64: tensor<4096x11008xf32>, %arg65: tensor<11008x4096xf32>, %arg66: tensor<4096xf32>, %arg67: tensor<4096x4096xf32>, %arg68: tensor<4096x4096xf32>, %arg69: tensor<4096xf32>, %arg70: tensor<4096x11008xf32>, %arg71: tensor<11008x4096xf32>, %arg72: tensor<4096xf32>, %arg73: tensor<4096x4096xf32>, %arg74: tensor<4096x4096xf32>, %arg75: tensor<4096xf32>, %arg76: tensor<4096x11008xf32>, %arg77: tensor<11008x4096xf32>, %arg78: tensor<4096xf32>, %arg79: tensor<4096x4096xf32>, %arg80: tensor<4096x4096xf32>, %arg81: tensor<4096xf32>, %arg82: tensor<4096x11008xf32>, %arg83: tensor<11008x4096xf32>, %arg84: tensor<4096xf32>, %arg85: tensor<4096x4096xf32>, %arg86: tensor<4096x4096xf32>, %arg87: tensor<4096xf32>, %arg88: tensor<4096x11008xf32>, %arg89: tensor<11008x4096xf32>, %arg90: tensor<4096xf32>, %arg91: tensor<4096x4096xf32>, %arg92: tensor<4096x4096xf32>, %arg93: tensor<4096xf32>, %arg94: tensor<4096x11008xf32>, %arg95: tensor<11008x4096xf32>, %arg96: tensor<4096xf32>, %arg97: tensor<4096x4096xf32>, %arg98: tensor<4096x4096xf32>, %arg99: tensor<4096xf32>, %arg100: tensor<4096x11008xf32>, %arg101: tensor<11008x4096xf32>, %arg102: tensor<4096xf32>, %arg103: tensor<4096x4096xf32>, %arg104: tensor<4096x4096xf32>, %arg105: tensor<4096xf32>, %arg106: tensor<4096x11008xf32>, %arg107: tensor<11008x4096xf32>, %arg108: tensor<4096xf32>, %arg109: tensor<4096x4096xf32>, %arg110: tensor<4096x4096xf32>, %arg111: tensor<4096xf32>, %arg112: tensor<4096x11008xf32>, %arg113: tensor<11008x4096xf32>, %arg114: tensor<4096xf32>, %arg115: tensor<4096x4096xf32>, %arg116: tensor<4096x4096xf32>, %arg117: tensor<4096xf32>, %arg118: tensor<4096x11008xf32>, %arg119: tensor<11008x4096xf32>, %arg120: tensor<4096xf32>, %arg121: tensor<4096x4096xf32>, %arg122: tensor<4096x4096xf32>, %arg123: tensor<4096xf32>, %arg124: tensor<4096x11008xf32>, %arg125: tensor<11008x4096xf32>, %arg126: tensor<4096xf32>, %arg127: tensor<4096x4096xf32>, %arg128: tensor<4096x4096xf32>, %arg129: tensor<4096xf32>, %arg130: tensor<4096x11008xf32>, %arg131: tensor<11008x4096xf32>, %arg132: tensor<4096xf32>, %arg133: tensor<4096x4096xf32>, %arg134: tensor<4096x4096xf32>, %arg135: tensor<4096xf32>, %arg136: tensor<4096x11008xf32>, %arg137: tensor<11008x4096xf32>, %arg138: tensor<4096xf32>, %arg139: tensor<4096x4096xf32>, %arg140: tensor<4096x4096xf32>, %arg141: tensor<4096xf32>, %arg142: tensor<4096x11008xf32>, %arg143: tensor<11008x4096xf32>, %arg144: tensor<4096xf32>, %arg145: tensor<4096x4096xf32>, %arg146: tensor<4096x4096xf32>, %arg147: tensor<4096xf32>, %arg148: tensor<4096x11008xf32>, %arg149: tensor<11008x4096xf32>, %arg150: tensor<4096xf32>, %arg151: tensor<4096x4096xf32>, %arg152: tensor<4096x4096xf32>, %arg153: tensor<4096xf32>, %arg154: tensor<4096x11008xf32>, %arg155: tensor<11008x4096xf32>, %arg156: tensor<4096xf32>, %arg157: tensor<4096x4096xf32>, %arg158: tensor<4096x4096xf32>, %arg159: tensor<4096xf32>, %arg160: tensor<4096x11008xf32>, %arg161: tensor<11008x4096xf32>, %arg162: tensor<4096xf32>, %arg163: tensor<4096x4096xf32>, %arg164: tensor<4096x4096xf32>, %arg165: tensor<4096xf32>, %arg166: tensor<4096x11008xf32>, %arg167: tensor<11008x4096xf32>, %arg168: tensor<4096xf32>, %arg169: tensor<4096x4096xf32>, %arg170: tensor<4096x4096xf32>, %arg171: tensor<4096xf32>, %arg172: tensor<4096x11008xf32>, %arg173: tensor<11008x4096xf32>, %arg174: tensor<4096xf32>, %arg175: tensor<4096x4096xf32>, %arg176: tensor<4096x4096xf32>, %arg177: tensor<4096xf32>, %arg178: tensor<4096x11008xf32>, %arg179: tensor<11008x4096xf32>, %arg180: tensor<4096xf32>, %arg181: tensor<4096x4096xf32>, %arg182: tensor<4096x4096xf32>, %arg183: tensor<4096xf32>, %arg184: tensor<4096x11008xf32>, %arg185: tensor<11008x4096xf32>, %arg186: tensor<4096xf32>, %arg187: tensor<4096x4096xf32>, %arg188: tensor<4096x4096xf32>, %arg189: tensor<4096xf32>, %arg190: tensor<4096x11008xf32>, %arg191: tensor<11008x4096xf32>, %arg192: tensor<4096xf32>, %arg193: tensor<4096x4096xf32>, %arg194: tensor<4096x4096xf32>, %arg195: tensor<4096xf32>, %arg196: tensor<1xi64>, %arg197: tensor<32000x4096xf32>, %arg198: tensor<1xi64>, %arg199: tensor<i64>, %arg200: tensor<2304x32x128xf32>, %arg201: tensor<f32>, %arg202: tensor<4608x64xcomplex<f32>>, %arg203: tensor<4096x4096xf32>, %arg204: tensor<2304x32x128xf32>, %arg205: tensor<4096x4096xf32>, %arg206: tensor<11008x4096xf32>, %arg207: tensor<2304x32x128xf32>, %arg208: tensor<4096x4096xf32>, %arg209: tensor<2304x32x128xf32>, %arg210: tensor<4096x4096xf32>, %arg211: tensor<11008x4096xf32>, %arg212: tensor<2304x32x128xf32>, %arg213: tensor<4096x4096xf32>, %arg214: tensor<2304x32x128xf32>, %arg215: tensor<4096x4096xf32>, %arg216: tensor<11008x4096xf32>, %arg217: tensor<2304x32x128xf32>, %arg218: tensor<4096x4096xf32>, %arg219: tensor<2304x32x128xf32>, %arg220: tensor<4096x4096xf32>, %arg221: tensor<11008x4096xf32>, %arg222: tensor<2304x32x128xf32>, %arg223: tensor<4096x4096xf32>, %arg224: tensor<2304x32x128xf32>, %arg225: tensor<4096x4096xf32>, %arg226: tensor<11008x4096xf32>, %arg227: tensor<2304x32x128xf32>, %arg228: tensor<4096x4096xf32>, %arg229: tensor<2304x32x128xf32>, %arg230: tensor<4096x4096xf32>, %arg231: tensor<11008x4096xf32>, %arg232: tensor<2304x32x128xf32>, %arg233: tensor<4096x4096xf32>, %arg234: tensor<2304x32x128xf32>, %arg235: tensor<4096x4096xf32>, %arg236: tensor<11008x4096xf32>, %arg237: tensor<2304x32x128xf32>, %arg238: tensor<4096x4096xf32>, %arg239: tensor<2304x32x128xf32>, %arg240: tensor<4096x4096xf32>, %arg241: tensor<11008x4096xf32>, %arg242: tensor<2304x32x128xf32>, %arg243: tensor<4096x4096xf32>, %arg244: tensor<2304x32x128xf32>, %arg245: tensor<4096x4096xf32>, %arg246: tensor<11008x4096xf32>, %arg247: tensor<2304x32x128xf32>, %arg248: tensor<4096x4096xf32>, %arg249: tensor<2304x32x128xf32>, %arg250: tensor<4096x4096xf32>, %arg251: tensor<11008x4096xf32>, %arg252: tensor<2304x32x128xf32>, %arg253: tensor<4096x4096xf32>, %arg254: tensor<2304x32x128xf32>, %arg255: tensor<4096x4096xf32>, %arg256: tensor<11008x4096xf32>, %arg257: tensor<2304x32x128xf32>, %arg258: tensor<4096x4096xf32>, %arg259: tensor<2304x32x128xf32>, %arg260: tensor<4096x4096xf32>, %arg261: tensor<11008x4096xf32>, %arg262: tensor<2304x32x128xf32>, %arg263: tensor<4096x4096xf32>, %arg264: tensor<2304x32x128xf32>, %arg265: tensor<4096x4096xf32>, %arg266: tensor<11008x4096xf32>, %arg267: tensor<2304x32x128xf32>, %arg268: tensor<4096x4096xf32>, %arg269: tensor<2304x32x128xf32>, %arg270: tensor<4096x4096xf32>, %arg271: tensor<11008x4096xf32>, %arg272: tensor<2304x32x128xf32>, %arg273: tensor<4096x4096xf32>, %arg274: tensor<2304x32x128xf32>, %arg275: tensor<4096x4096xf32>, %arg276: tensor<11008x4096xf32>, %arg277: tensor<2304x32x128xf32>, %arg278: tensor<4096x4096xf32>, %arg279: tensor<2304x32x128xf32>, %arg280: tensor<4096x4096xf32>, %arg281: tensor<11008x4096xf32>, %arg282: tensor<2304x32x128xf32>, %arg283: tensor<4096x4096xf32>, %arg284: tensor<2304x32x128xf32>, %arg285: tensor<4096x4096xf32>, %arg286: tensor<11008x4096xf32>, %arg287: tensor<2304x32x128xf32>, %arg288: tensor<4096x4096xf32>, %arg289: tensor<2304x32x128xf32>, %arg290: tensor<4096x4096xf32>, %arg291: tensor<11008x4096xf32>, %arg292: tensor<2304x32x128xf32>, %arg293: tensor<4096x4096xf32>, %arg294: tensor<2304x32x128xf32>, %arg295: tensor<4096x4096xf32>, %arg296: tensor<11008x4096xf32>, %arg297: tensor<2304x32x128xf32>, %arg298: tensor<4096x4096xf32>, %arg299: tensor<2304x32x128xf32>, %arg300: tensor<4096x4096xf32>, %arg301: tensor<11008x4096xf32>, %arg302: tensor<2304x32x128xf32>, %arg303: tensor<4096x4096xf32>, %arg304: tensor<2304x32x128xf32>, %arg305: tensor<4096x4096xf32>, %arg306: tensor<11008x4096xf32>, %arg307: tensor<2304x32x128xf32>, %arg308: tensor<4096x4096xf32>, %arg309: tensor<2304x32x128xf32>, %arg310: tensor<4096x4096xf32>, %arg311: tensor<11008x4096xf32>, %arg312: tensor<2304x32x128xf32>, %arg313: tensor<4096x4096xf32>, %arg314: tensor<2304x32x128xf32>, %arg315: tensor<4096x4096xf32>, %arg316: tensor<11008x4096xf32>, %arg317: tensor<2304x32x128xf32>, %arg318: tensor<4096x4096xf32>, %arg319: tensor<2304x32x128xf32>, %arg320: tensor<4096x4096xf32>, %arg321: tensor<11008x4096xf32>, %arg322: tensor<2304x32x128xf32>, %arg323: tensor<4096x4096xf32>, %arg324: tensor<2304x32x128xf32>, %arg325: tensor<4096x4096xf32>, %arg326: tensor<11008x4096xf32>, %arg327: tensor<2304x32x128xf32>, %arg328: tensor<4096x4096xf32>, %arg329: tensor<2304x32x128xf32>, %arg330: tensor<4096x4096xf32>, %arg331: tensor<11008x4096xf32>, %arg332: tensor<2304x32x128xf32>, %arg333: tensor<4096x4096xf32>, %arg334: tensor<2304x32x128xf32>, %arg335: tensor<4096x4096xf32>, %arg336: tensor<11008x4096xf32>, %arg337: tensor<2304x32x128xf32>, %arg338: tensor<4096x4096xf32>, %arg339: tensor<2304x32x128xf32>, %arg340: tensor<4096x4096xf32>, %arg341: tensor<11008x4096xf32>, %arg342: tensor<2304x32x128xf32>, %arg343: tensor<4096x4096xf32>, %arg344: tensor<2304x32x128xf32>, %arg345: tensor<4096x4096xf32>, %arg346: tensor<11008x4096xf32>, %arg347: tensor<2304x32x128xf32>, %arg348: tensor<4096x4096xf32>, %arg349: tensor<2304x32x128xf32>, %arg350: tensor<4096x4096xf32>, %arg351: tensor<11008x4096xf32>, %arg352: tensor<2304x32x128xf32>, %arg353: tensor<4096x4096xf32>, %arg354: tensor<2304x32x128xf32>, %arg355: tensor<4096x4096xf32>, %arg356: tensor<11008x4096xf32>, %arg357: tensor<2304x32x128xf32>, %arg358: tensor<4096x4096xf32>, %arg359: tensor<2304x32x128xf32>, %arg360: tensor<4096x4096xf32>, %arg361: tensor<11008x4096xf32>) -> (tensor<1x32000xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>) {
    %0 = stablehlo.constant dense<2.44140625E-4> : tensor<1xf32>
    %1 = stablehlo.constant dense<2.000000e+00> : tensor<1x4096xf32>
    %2 = stablehlo.constant dense<0> : tensor<1xi64>
    %3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.convert %arg196 : (tensor<1xi64>) -> tensor<1xui32>
    %6 = "stablehlo.gather"(%arg197, %5) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 4096]> : tensor<2xi64>} : (tensor<32000x4096xf32>, tensor<1xui32>) -> tensor<1x4096xf32>
    %7 = stablehlo.power %6, %1 : tensor<1x4096xf32>
    %8 = stablehlo.reduce(%7 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %9 = stablehlo.multiply %8, %0 : tensor<1xf32>
    %10 = stablehlo.reshape %9 : (tensor<1xf32>) -> tensor<1x1xf32>
    %11 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %12 = stablehlo.add %10, %11 : tensor<1x1xf32>
    %13 = stablehlo.rsqrt %12 : tensor<1x1xf32>
    %14 = stablehlo.reshape %13 : (tensor<1x1xf32>) -> tensor<1xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %16 = stablehlo.multiply %6, %15 : tensor<1x4096xf32>
    %17 = stablehlo.reshape %arg195 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<1x4096xf32>
    %19 = stablehlo.transpose %arg205, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %20 = stablehlo.dot %18, %19, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %21 = stablehlo.reshape %20 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %22 = stablehlo.slice %21 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %23 = stablehlo.reshape %22 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %24 = stablehlo.slice %21 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %25 = stablehlo.reshape %24 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %26 = stablehlo.complex %23, %25 : tensor<1x32x64xcomplex<f32>>
    %27 = stablehlo.convert %arg198 : (tensor<1xi64>) -> tensor<1xui32>
    %28 = "stablehlo.gather"(%arg202, %27) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 64]> : tensor<2xi64>} : (tensor<4608x64xcomplex<f32>>, tensor<1xui32>) -> tensor<1x64xcomplex<f32>>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %30 = stablehlo.multiply %26, %29 : tensor<1x32x64xcomplex<f32>>
    %31 = stablehlo.real %30 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %32 = stablehlo.reshape %31 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %33 = stablehlo.imag %30 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %34 = stablehlo.reshape %33 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %35 = stablehlo.concatenate %32, %34, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %36 = stablehlo.reshape %35 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %37 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %38 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %39 = stablehlo.add %arg198, %38 : tensor<1xi64>
    %40 = stablehlo.select %37, %39, %arg198 : tensor<1xi1>, tensor<1xi64>
    %41 = stablehlo.reshape %40 : (tensor<1xi64>) -> tensor<1x1xi64>
    %42 = stablehlo.transpose %arg203, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %43 = stablehlo.dot %18, %42, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %44 = stablehlo.reshape %43 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %45 = stablehlo.slice %44 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %46 = stablehlo.reshape %45 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %47 = stablehlo.slice %44 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %48 = stablehlo.reshape %47 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %49 = stablehlo.complex %46, %48 : tensor<1x32x64xcomplex<f32>>
    %50 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %51 = stablehlo.multiply %49, %50 : tensor<1x32x64xcomplex<f32>>
    %52 = stablehlo.real %51 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %53 = stablehlo.reshape %52 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %54 = stablehlo.imag %51 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %55 = stablehlo.reshape %54 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %56 = stablehlo.concatenate %53, %55, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %57 = stablehlo.reshape %56 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %58 = "stablehlo.scatter"(%arg204, %41, %57) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %59 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %60 = "stablehlo.gather"(%58, %59) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %61 = stablehlo.transpose %60, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %62 = stablehlo.dot_general %36, %61, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %63 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %64 = stablehlo.divide %62, %63 : tensor<32x1x2049xf32>
    %65 = stablehlo.reduce(%64 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %66 = stablehlo.broadcast_in_dim %65, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %67 = stablehlo.subtract %64, %66 : tensor<32x1x2049xf32>
    %68 = stablehlo.exponential %67 : tensor<32x1x2049xf32>
    %69 = stablehlo.reduce(%68 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %70 = stablehlo.broadcast_in_dim %69, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %71 = stablehlo.divide %68, %70 : tensor<32x1x2049xf32>
    %72 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %73 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %74 = stablehlo.add %arg198, %73 : tensor<1xi64>
    %75 = stablehlo.select %72, %74, %arg198 : tensor<1xi1>, tensor<1xi64>
    %76 = stablehlo.reshape %75 : (tensor<1xi64>) -> tensor<1x1xi64>
    %77 = stablehlo.transpose %arg194, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %78 = stablehlo.dot %18, %77, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %79 = stablehlo.reshape %78 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %80 = "stablehlo.scatter"(%arg200, %76, %79) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %81 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %82 = "stablehlo.gather"(%80, %81) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %83 = stablehlo.transpose %82, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %84 = stablehlo.dot_general %71, %83, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %85 = stablehlo.reshape %84 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %86 = stablehlo.transpose %arg193, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %87 = stablehlo.dot %85, %86, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %88 = stablehlo.add %6, %87 : tensor<1x4096xf32>
    %89 = stablehlo.power %88, %1 : tensor<1x4096xf32>
    %90 = stablehlo.reduce(%89 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %91 = stablehlo.multiply %90, %0 : tensor<1xf32>
    %92 = stablehlo.reshape %91 : (tensor<1xf32>) -> tensor<1x1xf32>
    %93 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %94 = stablehlo.add %92, %93 : tensor<1x1xf32>
    %95 = stablehlo.rsqrt %94 : tensor<1x1xf32>
    %96 = stablehlo.reshape %95 : (tensor<1x1xf32>) -> tensor<1xf32>
    %97 = stablehlo.broadcast_in_dim %96, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %98 = stablehlo.multiply %88, %97 : tensor<1x4096xf32>
    %99 = stablehlo.reshape %arg192 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %100 = stablehlo.multiply %98, %99 : tensor<1x4096xf32>
    %101 = stablehlo.transpose %arg206, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %102 = stablehlo.dot %100, %101, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %103 = stablehlo.logistic %102 : tensor<1x11008xf32>
    %104 = stablehlo.multiply %102, %103 : tensor<1x11008xf32>
    %105 = stablehlo.transpose %arg191, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %106 = stablehlo.dot %100, %105, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %107 = stablehlo.multiply %104, %106 : tensor<1x11008xf32>
    %108 = stablehlo.transpose %arg190, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %109 = stablehlo.dot %107, %108, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %110 = stablehlo.add %88, %109 : tensor<1x4096xf32>
    %111 = stablehlo.power %110, %1 : tensor<1x4096xf32>
    %112 = stablehlo.reduce(%111 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %113 = stablehlo.multiply %112, %0 : tensor<1xf32>
    %114 = stablehlo.reshape %113 : (tensor<1xf32>) -> tensor<1x1xf32>
    %115 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %116 = stablehlo.add %114, %115 : tensor<1x1xf32>
    %117 = stablehlo.rsqrt %116 : tensor<1x1xf32>
    %118 = stablehlo.reshape %117 : (tensor<1x1xf32>) -> tensor<1xf32>
    %119 = stablehlo.broadcast_in_dim %118, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %120 = stablehlo.multiply %110, %119 : tensor<1x4096xf32>
    %121 = stablehlo.reshape %arg189 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %122 = stablehlo.multiply %120, %121 : tensor<1x4096xf32>
    %123 = stablehlo.transpose %arg210, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %124 = stablehlo.dot %122, %123, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %125 = stablehlo.reshape %124 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %126 = stablehlo.slice %125 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %127 = stablehlo.reshape %126 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %128 = stablehlo.slice %125 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %129 = stablehlo.reshape %128 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %130 = stablehlo.complex %127, %129 : tensor<1x32x64xcomplex<f32>>
    %131 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %132 = stablehlo.multiply %130, %131 : tensor<1x32x64xcomplex<f32>>
    %133 = stablehlo.real %132 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %134 = stablehlo.reshape %133 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %135 = stablehlo.imag %132 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %136 = stablehlo.reshape %135 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %137 = stablehlo.concatenate %134, %136, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %138 = stablehlo.reshape %137 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %139 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %140 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %141 = stablehlo.add %arg198, %140 : tensor<1xi64>
    %142 = stablehlo.select %139, %141, %arg198 : tensor<1xi1>, tensor<1xi64>
    %143 = stablehlo.reshape %142 : (tensor<1xi64>) -> tensor<1x1xi64>
    %144 = stablehlo.transpose %arg208, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %145 = stablehlo.dot %122, %144, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %146 = stablehlo.reshape %145 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %147 = stablehlo.slice %146 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %148 = stablehlo.reshape %147 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %149 = stablehlo.slice %146 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %150 = stablehlo.reshape %149 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %151 = stablehlo.complex %148, %150 : tensor<1x32x64xcomplex<f32>>
    %152 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %153 = stablehlo.multiply %151, %152 : tensor<1x32x64xcomplex<f32>>
    %154 = stablehlo.real %153 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %155 = stablehlo.reshape %154 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %156 = stablehlo.imag %153 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %157 = stablehlo.reshape %156 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %158 = stablehlo.concatenate %155, %157, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %159 = stablehlo.reshape %158 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %160 = "stablehlo.scatter"(%arg209, %143, %159) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %161 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %162 = "stablehlo.gather"(%160, %161) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %163 = stablehlo.transpose %162, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %164 = stablehlo.dot_general %138, %163, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %165 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %166 = stablehlo.divide %164, %165 : tensor<32x1x2049xf32>
    %167 = stablehlo.reduce(%166 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %168 = stablehlo.broadcast_in_dim %167, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %169 = stablehlo.subtract %166, %168 : tensor<32x1x2049xf32>
    %170 = stablehlo.exponential %169 : tensor<32x1x2049xf32>
    %171 = stablehlo.reduce(%170 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %172 = stablehlo.broadcast_in_dim %171, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %173 = stablehlo.divide %170, %172 : tensor<32x1x2049xf32>
    %174 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %175 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %176 = stablehlo.add %arg198, %175 : tensor<1xi64>
    %177 = stablehlo.select %174, %176, %arg198 : tensor<1xi1>, tensor<1xi64>
    %178 = stablehlo.reshape %177 : (tensor<1xi64>) -> tensor<1x1xi64>
    %179 = stablehlo.transpose %arg188, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %180 = stablehlo.dot %122, %179, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %181 = stablehlo.reshape %180 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %182 = "stablehlo.scatter"(%arg207, %178, %181) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %183 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %184 = "stablehlo.gather"(%182, %183) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %185 = stablehlo.transpose %184, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %186 = stablehlo.dot_general %173, %185, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %187 = stablehlo.reshape %186 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %188 = stablehlo.transpose %arg187, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %189 = stablehlo.dot %187, %188, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %190 = stablehlo.add %110, %189 : tensor<1x4096xf32>
    %191 = stablehlo.power %190, %1 : tensor<1x4096xf32>
    %192 = stablehlo.reduce(%191 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %193 = stablehlo.multiply %192, %0 : tensor<1xf32>
    %194 = stablehlo.reshape %193 : (tensor<1xf32>) -> tensor<1x1xf32>
    %195 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %196 = stablehlo.add %194, %195 : tensor<1x1xf32>
    %197 = stablehlo.rsqrt %196 : tensor<1x1xf32>
    %198 = stablehlo.reshape %197 : (tensor<1x1xf32>) -> tensor<1xf32>
    %199 = stablehlo.broadcast_in_dim %198, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %200 = stablehlo.multiply %190, %199 : tensor<1x4096xf32>
    %201 = stablehlo.reshape %arg186 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %202 = stablehlo.multiply %200, %201 : tensor<1x4096xf32>
    %203 = stablehlo.transpose %arg211, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %204 = stablehlo.dot %202, %203, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %205 = stablehlo.logistic %204 : tensor<1x11008xf32>
    %206 = stablehlo.multiply %204, %205 : tensor<1x11008xf32>
    %207 = stablehlo.transpose %arg185, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %208 = stablehlo.dot %202, %207, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %209 = stablehlo.multiply %206, %208 : tensor<1x11008xf32>
    %210 = stablehlo.transpose %arg184, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %211 = stablehlo.dot %209, %210, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %212 = stablehlo.add %190, %211 : tensor<1x4096xf32>
    %213 = stablehlo.power %212, %1 : tensor<1x4096xf32>
    %214 = stablehlo.reduce(%213 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %215 = stablehlo.multiply %214, %0 : tensor<1xf32>
    %216 = stablehlo.reshape %215 : (tensor<1xf32>) -> tensor<1x1xf32>
    %217 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %218 = stablehlo.add %216, %217 : tensor<1x1xf32>
    %219 = stablehlo.rsqrt %218 : tensor<1x1xf32>
    %220 = stablehlo.reshape %219 : (tensor<1x1xf32>) -> tensor<1xf32>
    %221 = stablehlo.broadcast_in_dim %220, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %222 = stablehlo.multiply %212, %221 : tensor<1x4096xf32>
    %223 = stablehlo.reshape %arg183 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %224 = stablehlo.multiply %222, %223 : tensor<1x4096xf32>
    %225 = stablehlo.transpose %arg215, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %226 = stablehlo.dot %224, %225, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %227 = stablehlo.reshape %226 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %228 = stablehlo.slice %227 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %229 = stablehlo.reshape %228 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %230 = stablehlo.slice %227 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %231 = stablehlo.reshape %230 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %232 = stablehlo.complex %229, %231 : tensor<1x32x64xcomplex<f32>>
    %233 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %234 = stablehlo.multiply %232, %233 : tensor<1x32x64xcomplex<f32>>
    %235 = stablehlo.real %234 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %236 = stablehlo.reshape %235 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %237 = stablehlo.imag %234 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %238 = stablehlo.reshape %237 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %239 = stablehlo.concatenate %236, %238, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %240 = stablehlo.reshape %239 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %241 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %242 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %243 = stablehlo.add %arg198, %242 : tensor<1xi64>
    %244 = stablehlo.select %241, %243, %arg198 : tensor<1xi1>, tensor<1xi64>
    %245 = stablehlo.reshape %244 : (tensor<1xi64>) -> tensor<1x1xi64>
    %246 = stablehlo.transpose %arg213, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %247 = stablehlo.dot %224, %246, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %248 = stablehlo.reshape %247 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %249 = stablehlo.slice %248 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %250 = stablehlo.reshape %249 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %251 = stablehlo.slice %248 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %252 = stablehlo.reshape %251 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %253 = stablehlo.complex %250, %252 : tensor<1x32x64xcomplex<f32>>
    %254 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %255 = stablehlo.multiply %253, %254 : tensor<1x32x64xcomplex<f32>>
    %256 = stablehlo.real %255 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %257 = stablehlo.reshape %256 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %258 = stablehlo.imag %255 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %259 = stablehlo.reshape %258 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %260 = stablehlo.concatenate %257, %259, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %261 = stablehlo.reshape %260 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %262 = "stablehlo.scatter"(%arg214, %245, %261) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %263 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %264 = "stablehlo.gather"(%262, %263) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %265 = stablehlo.transpose %264, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %266 = stablehlo.dot_general %240, %265, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %267 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %268 = stablehlo.divide %266, %267 : tensor<32x1x2049xf32>
    %269 = stablehlo.reduce(%268 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %270 = stablehlo.broadcast_in_dim %269, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %271 = stablehlo.subtract %268, %270 : tensor<32x1x2049xf32>
    %272 = stablehlo.exponential %271 : tensor<32x1x2049xf32>
    %273 = stablehlo.reduce(%272 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %274 = stablehlo.broadcast_in_dim %273, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %275 = stablehlo.divide %272, %274 : tensor<32x1x2049xf32>
    %276 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %277 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %278 = stablehlo.add %arg198, %277 : tensor<1xi64>
    %279 = stablehlo.select %276, %278, %arg198 : tensor<1xi1>, tensor<1xi64>
    %280 = stablehlo.reshape %279 : (tensor<1xi64>) -> tensor<1x1xi64>
    %281 = stablehlo.transpose %arg182, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %282 = stablehlo.dot %224, %281, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %283 = stablehlo.reshape %282 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %284 = "stablehlo.scatter"(%arg212, %280, %283) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %285 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %286 = "stablehlo.gather"(%284, %285) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %287 = stablehlo.transpose %286, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %288 = stablehlo.dot_general %275, %287, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %289 = stablehlo.reshape %288 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %290 = stablehlo.transpose %arg181, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %291 = stablehlo.dot %289, %290, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %292 = stablehlo.add %212, %291 : tensor<1x4096xf32>
    %293 = stablehlo.power %292, %1 : tensor<1x4096xf32>
    %294 = stablehlo.reduce(%293 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %295 = stablehlo.multiply %294, %0 : tensor<1xf32>
    %296 = stablehlo.reshape %295 : (tensor<1xf32>) -> tensor<1x1xf32>
    %297 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %298 = stablehlo.add %296, %297 : tensor<1x1xf32>
    %299 = stablehlo.rsqrt %298 : tensor<1x1xf32>
    %300 = stablehlo.reshape %299 : (tensor<1x1xf32>) -> tensor<1xf32>
    %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %302 = stablehlo.multiply %292, %301 : tensor<1x4096xf32>
    %303 = stablehlo.reshape %arg180 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %304 = stablehlo.multiply %302, %303 : tensor<1x4096xf32>
    %305 = stablehlo.transpose %arg216, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %306 = stablehlo.dot %304, %305, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %307 = stablehlo.logistic %306 : tensor<1x11008xf32>
    %308 = stablehlo.multiply %306, %307 : tensor<1x11008xf32>
    %309 = stablehlo.transpose %arg179, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %310 = stablehlo.dot %304, %309, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %311 = stablehlo.multiply %308, %310 : tensor<1x11008xf32>
    %312 = stablehlo.transpose %arg178, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %313 = stablehlo.dot %311, %312, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %314 = stablehlo.add %292, %313 : tensor<1x4096xf32>
    %315 = stablehlo.power %314, %1 : tensor<1x4096xf32>
    %316 = stablehlo.reduce(%315 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %317 = stablehlo.multiply %316, %0 : tensor<1xf32>
    %318 = stablehlo.reshape %317 : (tensor<1xf32>) -> tensor<1x1xf32>
    %319 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %320 = stablehlo.add %318, %319 : tensor<1x1xf32>
    %321 = stablehlo.rsqrt %320 : tensor<1x1xf32>
    %322 = stablehlo.reshape %321 : (tensor<1x1xf32>) -> tensor<1xf32>
    %323 = stablehlo.broadcast_in_dim %322, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %324 = stablehlo.multiply %314, %323 : tensor<1x4096xf32>
    %325 = stablehlo.reshape %arg177 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %326 = stablehlo.multiply %324, %325 : tensor<1x4096xf32>
    %327 = stablehlo.transpose %arg220, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %328 = stablehlo.dot %326, %327, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %329 = stablehlo.reshape %328 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %330 = stablehlo.slice %329 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %331 = stablehlo.reshape %330 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %332 = stablehlo.slice %329 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %333 = stablehlo.reshape %332 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %334 = stablehlo.complex %331, %333 : tensor<1x32x64xcomplex<f32>>
    %335 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %336 = stablehlo.multiply %334, %335 : tensor<1x32x64xcomplex<f32>>
    %337 = stablehlo.real %336 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %338 = stablehlo.reshape %337 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %339 = stablehlo.imag %336 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %340 = stablehlo.reshape %339 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %341 = stablehlo.concatenate %338, %340, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %342 = stablehlo.reshape %341 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %343 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %344 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %345 = stablehlo.add %arg198, %344 : tensor<1xi64>
    %346 = stablehlo.select %343, %345, %arg198 : tensor<1xi1>, tensor<1xi64>
    %347 = stablehlo.reshape %346 : (tensor<1xi64>) -> tensor<1x1xi64>
    %348 = stablehlo.transpose %arg218, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %349 = stablehlo.dot %326, %348, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %350 = stablehlo.reshape %349 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %351 = stablehlo.slice %350 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %352 = stablehlo.reshape %351 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %353 = stablehlo.slice %350 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %354 = stablehlo.reshape %353 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %355 = stablehlo.complex %352, %354 : tensor<1x32x64xcomplex<f32>>
    %356 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %357 = stablehlo.multiply %355, %356 : tensor<1x32x64xcomplex<f32>>
    %358 = stablehlo.real %357 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %359 = stablehlo.reshape %358 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %360 = stablehlo.imag %357 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %361 = stablehlo.reshape %360 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %362 = stablehlo.concatenate %359, %361, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %363 = stablehlo.reshape %362 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %364 = "stablehlo.scatter"(%arg219, %347, %363) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %365 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %366 = "stablehlo.gather"(%364, %365) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %367 = stablehlo.transpose %366, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %368 = stablehlo.dot_general %342, %367, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %369 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %370 = stablehlo.divide %368, %369 : tensor<32x1x2049xf32>
    %371 = stablehlo.reduce(%370 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %372 = stablehlo.broadcast_in_dim %371, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %373 = stablehlo.subtract %370, %372 : tensor<32x1x2049xf32>
    %374 = stablehlo.exponential %373 : tensor<32x1x2049xf32>
    %375 = stablehlo.reduce(%374 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %376 = stablehlo.broadcast_in_dim %375, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %377 = stablehlo.divide %374, %376 : tensor<32x1x2049xf32>
    %378 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %379 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %380 = stablehlo.add %arg198, %379 : tensor<1xi64>
    %381 = stablehlo.select %378, %380, %arg198 : tensor<1xi1>, tensor<1xi64>
    %382 = stablehlo.reshape %381 : (tensor<1xi64>) -> tensor<1x1xi64>
    %383 = stablehlo.transpose %arg176, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %384 = stablehlo.dot %326, %383, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %385 = stablehlo.reshape %384 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %386 = "stablehlo.scatter"(%arg217, %382, %385) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %387 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %388 = "stablehlo.gather"(%386, %387) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %389 = stablehlo.transpose %388, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %390 = stablehlo.dot_general %377, %389, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %391 = stablehlo.reshape %390 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %392 = stablehlo.transpose %arg175, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %393 = stablehlo.dot %391, %392, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %394 = stablehlo.add %314, %393 : tensor<1x4096xf32>
    %395 = stablehlo.power %394, %1 : tensor<1x4096xf32>
    %396 = stablehlo.reduce(%395 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %397 = stablehlo.multiply %396, %0 : tensor<1xf32>
    %398 = stablehlo.reshape %397 : (tensor<1xf32>) -> tensor<1x1xf32>
    %399 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %400 = stablehlo.add %398, %399 : tensor<1x1xf32>
    %401 = stablehlo.rsqrt %400 : tensor<1x1xf32>
    %402 = stablehlo.reshape %401 : (tensor<1x1xf32>) -> tensor<1xf32>
    %403 = stablehlo.broadcast_in_dim %402, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %404 = stablehlo.multiply %394, %403 : tensor<1x4096xf32>
    %405 = stablehlo.reshape %arg174 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %406 = stablehlo.multiply %404, %405 : tensor<1x4096xf32>
    %407 = stablehlo.transpose %arg221, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %408 = stablehlo.dot %406, %407, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %409 = stablehlo.logistic %408 : tensor<1x11008xf32>
    %410 = stablehlo.multiply %408, %409 : tensor<1x11008xf32>
    %411 = stablehlo.transpose %arg173, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %412 = stablehlo.dot %406, %411, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %413 = stablehlo.multiply %410, %412 : tensor<1x11008xf32>
    %414 = stablehlo.transpose %arg172, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %415 = stablehlo.dot %413, %414, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %416 = stablehlo.add %394, %415 : tensor<1x4096xf32>
    %417 = stablehlo.power %416, %1 : tensor<1x4096xf32>
    %418 = stablehlo.reduce(%417 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %419 = stablehlo.multiply %418, %0 : tensor<1xf32>
    %420 = stablehlo.reshape %419 : (tensor<1xf32>) -> tensor<1x1xf32>
    %421 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %422 = stablehlo.add %420, %421 : tensor<1x1xf32>
    %423 = stablehlo.rsqrt %422 : tensor<1x1xf32>
    %424 = stablehlo.reshape %423 : (tensor<1x1xf32>) -> tensor<1xf32>
    %425 = stablehlo.broadcast_in_dim %424, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %426 = stablehlo.multiply %416, %425 : tensor<1x4096xf32>
    %427 = stablehlo.reshape %arg171 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %428 = stablehlo.multiply %426, %427 : tensor<1x4096xf32>
    %429 = stablehlo.transpose %arg225, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %430 = stablehlo.dot %428, %429, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %431 = stablehlo.reshape %430 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %432 = stablehlo.slice %431 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %433 = stablehlo.reshape %432 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %434 = stablehlo.slice %431 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %435 = stablehlo.reshape %434 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %436 = stablehlo.complex %433, %435 : tensor<1x32x64xcomplex<f32>>
    %437 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %438 = stablehlo.multiply %436, %437 : tensor<1x32x64xcomplex<f32>>
    %439 = stablehlo.real %438 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %440 = stablehlo.reshape %439 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %441 = stablehlo.imag %438 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %442 = stablehlo.reshape %441 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %443 = stablehlo.concatenate %440, %442, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %444 = stablehlo.reshape %443 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %445 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %446 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %447 = stablehlo.add %arg198, %446 : tensor<1xi64>
    %448 = stablehlo.select %445, %447, %arg198 : tensor<1xi1>, tensor<1xi64>
    %449 = stablehlo.reshape %448 : (tensor<1xi64>) -> tensor<1x1xi64>
    %450 = stablehlo.transpose %arg223, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %451 = stablehlo.dot %428, %450, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %452 = stablehlo.reshape %451 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %453 = stablehlo.slice %452 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %454 = stablehlo.reshape %453 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %455 = stablehlo.slice %452 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %456 = stablehlo.reshape %455 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %457 = stablehlo.complex %454, %456 : tensor<1x32x64xcomplex<f32>>
    %458 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %459 = stablehlo.multiply %457, %458 : tensor<1x32x64xcomplex<f32>>
    %460 = stablehlo.real %459 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %461 = stablehlo.reshape %460 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %462 = stablehlo.imag %459 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %463 = stablehlo.reshape %462 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %464 = stablehlo.concatenate %461, %463, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %465 = stablehlo.reshape %464 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %466 = "stablehlo.scatter"(%arg224, %449, %465) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %467 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %468 = "stablehlo.gather"(%466, %467) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %469 = stablehlo.transpose %468, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %470 = stablehlo.dot_general %444, %469, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %471 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %472 = stablehlo.divide %470, %471 : tensor<32x1x2049xf32>
    %473 = stablehlo.reduce(%472 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %474 = stablehlo.broadcast_in_dim %473, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %475 = stablehlo.subtract %472, %474 : tensor<32x1x2049xf32>
    %476 = stablehlo.exponential %475 : tensor<32x1x2049xf32>
    %477 = stablehlo.reduce(%476 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %478 = stablehlo.broadcast_in_dim %477, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %479 = stablehlo.divide %476, %478 : tensor<32x1x2049xf32>
    %480 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %481 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %482 = stablehlo.add %arg198, %481 : tensor<1xi64>
    %483 = stablehlo.select %480, %482, %arg198 : tensor<1xi1>, tensor<1xi64>
    %484 = stablehlo.reshape %483 : (tensor<1xi64>) -> tensor<1x1xi64>
    %485 = stablehlo.transpose %arg170, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %486 = stablehlo.dot %428, %485, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %487 = stablehlo.reshape %486 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %488 = "stablehlo.scatter"(%arg222, %484, %487) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %489 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %490 = "stablehlo.gather"(%488, %489) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %491 = stablehlo.transpose %490, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %492 = stablehlo.dot_general %479, %491, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %493 = stablehlo.reshape %492 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %494 = stablehlo.transpose %arg169, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %495 = stablehlo.dot %493, %494, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %496 = stablehlo.add %416, %495 : tensor<1x4096xf32>
    %497 = stablehlo.power %496, %1 : tensor<1x4096xf32>
    %498 = stablehlo.reduce(%497 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %499 = stablehlo.multiply %498, %0 : tensor<1xf32>
    %500 = stablehlo.reshape %499 : (tensor<1xf32>) -> tensor<1x1xf32>
    %501 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %502 = stablehlo.add %500, %501 : tensor<1x1xf32>
    %503 = stablehlo.rsqrt %502 : tensor<1x1xf32>
    %504 = stablehlo.reshape %503 : (tensor<1x1xf32>) -> tensor<1xf32>
    %505 = stablehlo.broadcast_in_dim %504, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %506 = stablehlo.multiply %496, %505 : tensor<1x4096xf32>
    %507 = stablehlo.reshape %arg168 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %508 = stablehlo.multiply %506, %507 : tensor<1x4096xf32>
    %509 = stablehlo.transpose %arg226, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %510 = stablehlo.dot %508, %509, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %511 = stablehlo.logistic %510 : tensor<1x11008xf32>
    %512 = stablehlo.multiply %510, %511 : tensor<1x11008xf32>
    %513 = stablehlo.transpose %arg167, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %514 = stablehlo.dot %508, %513, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %515 = stablehlo.multiply %512, %514 : tensor<1x11008xf32>
    %516 = stablehlo.transpose %arg166, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %517 = stablehlo.dot %515, %516, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %518 = stablehlo.add %496, %517 : tensor<1x4096xf32>
    %519 = stablehlo.power %518, %1 : tensor<1x4096xf32>
    %520 = stablehlo.reduce(%519 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %521 = stablehlo.multiply %520, %0 : tensor<1xf32>
    %522 = stablehlo.reshape %521 : (tensor<1xf32>) -> tensor<1x1xf32>
    %523 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %524 = stablehlo.add %522, %523 : tensor<1x1xf32>
    %525 = stablehlo.rsqrt %524 : tensor<1x1xf32>
    %526 = stablehlo.reshape %525 : (tensor<1x1xf32>) -> tensor<1xf32>
    %527 = stablehlo.broadcast_in_dim %526, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %528 = stablehlo.multiply %518, %527 : tensor<1x4096xf32>
    %529 = stablehlo.reshape %arg165 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %530 = stablehlo.multiply %528, %529 : tensor<1x4096xf32>
    %531 = stablehlo.transpose %arg230, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %532 = stablehlo.dot %530, %531, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %533 = stablehlo.reshape %532 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %534 = stablehlo.slice %533 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %535 = stablehlo.reshape %534 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %536 = stablehlo.slice %533 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %537 = stablehlo.reshape %536 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %538 = stablehlo.complex %535, %537 : tensor<1x32x64xcomplex<f32>>
    %539 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %540 = stablehlo.multiply %538, %539 : tensor<1x32x64xcomplex<f32>>
    %541 = stablehlo.real %540 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %542 = stablehlo.reshape %541 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %543 = stablehlo.imag %540 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %544 = stablehlo.reshape %543 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %545 = stablehlo.concatenate %542, %544, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %546 = stablehlo.reshape %545 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %547 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %548 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %549 = stablehlo.add %arg198, %548 : tensor<1xi64>
    %550 = stablehlo.select %547, %549, %arg198 : tensor<1xi1>, tensor<1xi64>
    %551 = stablehlo.reshape %550 : (tensor<1xi64>) -> tensor<1x1xi64>
    %552 = stablehlo.transpose %arg228, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %553 = stablehlo.dot %530, %552, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %554 = stablehlo.reshape %553 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %555 = stablehlo.slice %554 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %556 = stablehlo.reshape %555 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %557 = stablehlo.slice %554 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %558 = stablehlo.reshape %557 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %559 = stablehlo.complex %556, %558 : tensor<1x32x64xcomplex<f32>>
    %560 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %561 = stablehlo.multiply %559, %560 : tensor<1x32x64xcomplex<f32>>
    %562 = stablehlo.real %561 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %563 = stablehlo.reshape %562 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %564 = stablehlo.imag %561 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %565 = stablehlo.reshape %564 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %566 = stablehlo.concatenate %563, %565, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %567 = stablehlo.reshape %566 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %568 = "stablehlo.scatter"(%arg229, %551, %567) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %569 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %570 = "stablehlo.gather"(%568, %569) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %571 = stablehlo.transpose %570, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %572 = stablehlo.dot_general %546, %571, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %573 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %574 = stablehlo.divide %572, %573 : tensor<32x1x2049xf32>
    %575 = stablehlo.reduce(%574 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %576 = stablehlo.broadcast_in_dim %575, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %577 = stablehlo.subtract %574, %576 : tensor<32x1x2049xf32>
    %578 = stablehlo.exponential %577 : tensor<32x1x2049xf32>
    %579 = stablehlo.reduce(%578 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %580 = stablehlo.broadcast_in_dim %579, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %581 = stablehlo.divide %578, %580 : tensor<32x1x2049xf32>
    %582 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %583 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %584 = stablehlo.add %arg198, %583 : tensor<1xi64>
    %585 = stablehlo.select %582, %584, %arg198 : tensor<1xi1>, tensor<1xi64>
    %586 = stablehlo.reshape %585 : (tensor<1xi64>) -> tensor<1x1xi64>
    %587 = stablehlo.transpose %arg164, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %588 = stablehlo.dot %530, %587, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %589 = stablehlo.reshape %588 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %590 = "stablehlo.scatter"(%arg227, %586, %589) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %591 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %592 = "stablehlo.gather"(%590, %591) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %593 = stablehlo.transpose %592, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %594 = stablehlo.dot_general %581, %593, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %595 = stablehlo.reshape %594 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %596 = stablehlo.transpose %arg163, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %597 = stablehlo.dot %595, %596, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %598 = stablehlo.add %518, %597 : tensor<1x4096xf32>
    %599 = stablehlo.power %598, %1 : tensor<1x4096xf32>
    %600 = stablehlo.reduce(%599 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %601 = stablehlo.multiply %600, %0 : tensor<1xf32>
    %602 = stablehlo.reshape %601 : (tensor<1xf32>) -> tensor<1x1xf32>
    %603 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %604 = stablehlo.add %602, %603 : tensor<1x1xf32>
    %605 = stablehlo.rsqrt %604 : tensor<1x1xf32>
    %606 = stablehlo.reshape %605 : (tensor<1x1xf32>) -> tensor<1xf32>
    %607 = stablehlo.broadcast_in_dim %606, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %608 = stablehlo.multiply %598, %607 : tensor<1x4096xf32>
    %609 = stablehlo.reshape %arg162 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %610 = stablehlo.multiply %608, %609 : tensor<1x4096xf32>
    %611 = stablehlo.transpose %arg231, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %612 = stablehlo.dot %610, %611, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %613 = stablehlo.logistic %612 : tensor<1x11008xf32>
    %614 = stablehlo.multiply %612, %613 : tensor<1x11008xf32>
    %615 = stablehlo.transpose %arg161, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %616 = stablehlo.dot %610, %615, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %617 = stablehlo.multiply %614, %616 : tensor<1x11008xf32>
    %618 = stablehlo.transpose %arg160, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %619 = stablehlo.dot %617, %618, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %620 = stablehlo.add %598, %619 : tensor<1x4096xf32>
    %621 = stablehlo.power %620, %1 : tensor<1x4096xf32>
    %622 = stablehlo.reduce(%621 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %623 = stablehlo.multiply %622, %0 : tensor<1xf32>
    %624 = stablehlo.reshape %623 : (tensor<1xf32>) -> tensor<1x1xf32>
    %625 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %626 = stablehlo.add %624, %625 : tensor<1x1xf32>
    %627 = stablehlo.rsqrt %626 : tensor<1x1xf32>
    %628 = stablehlo.reshape %627 : (tensor<1x1xf32>) -> tensor<1xf32>
    %629 = stablehlo.broadcast_in_dim %628, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %630 = stablehlo.multiply %620, %629 : tensor<1x4096xf32>
    %631 = stablehlo.reshape %arg159 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %632 = stablehlo.multiply %630, %631 : tensor<1x4096xf32>
    %633 = stablehlo.transpose %arg235, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %634 = stablehlo.dot %632, %633, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %635 = stablehlo.reshape %634 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %636 = stablehlo.slice %635 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %637 = stablehlo.reshape %636 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %638 = stablehlo.slice %635 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %639 = stablehlo.reshape %638 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %640 = stablehlo.complex %637, %639 : tensor<1x32x64xcomplex<f32>>
    %641 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %642 = stablehlo.multiply %640, %641 : tensor<1x32x64xcomplex<f32>>
    %643 = stablehlo.real %642 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %644 = stablehlo.reshape %643 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %645 = stablehlo.imag %642 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %646 = stablehlo.reshape %645 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %647 = stablehlo.concatenate %644, %646, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %648 = stablehlo.reshape %647 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %649 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %650 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %651 = stablehlo.add %arg198, %650 : tensor<1xi64>
    %652 = stablehlo.select %649, %651, %arg198 : tensor<1xi1>, tensor<1xi64>
    %653 = stablehlo.reshape %652 : (tensor<1xi64>) -> tensor<1x1xi64>
    %654 = stablehlo.transpose %arg233, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %655 = stablehlo.dot %632, %654, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %656 = stablehlo.reshape %655 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %657 = stablehlo.slice %656 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %658 = stablehlo.reshape %657 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %659 = stablehlo.slice %656 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %660 = stablehlo.reshape %659 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %661 = stablehlo.complex %658, %660 : tensor<1x32x64xcomplex<f32>>
    %662 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %663 = stablehlo.multiply %661, %662 : tensor<1x32x64xcomplex<f32>>
    %664 = stablehlo.real %663 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %665 = stablehlo.reshape %664 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %666 = stablehlo.imag %663 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %667 = stablehlo.reshape %666 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %668 = stablehlo.concatenate %665, %667, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %669 = stablehlo.reshape %668 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %670 = "stablehlo.scatter"(%arg234, %653, %669) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %671 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %672 = "stablehlo.gather"(%670, %671) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %673 = stablehlo.transpose %672, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %674 = stablehlo.dot_general %648, %673, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %675 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %676 = stablehlo.divide %674, %675 : tensor<32x1x2049xf32>
    %677 = stablehlo.reduce(%676 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %678 = stablehlo.broadcast_in_dim %677, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %679 = stablehlo.subtract %676, %678 : tensor<32x1x2049xf32>
    %680 = stablehlo.exponential %679 : tensor<32x1x2049xf32>
    %681 = stablehlo.reduce(%680 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %682 = stablehlo.broadcast_in_dim %681, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %683 = stablehlo.divide %680, %682 : tensor<32x1x2049xf32>
    %684 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %685 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %686 = stablehlo.add %arg198, %685 : tensor<1xi64>
    %687 = stablehlo.select %684, %686, %arg198 : tensor<1xi1>, tensor<1xi64>
    %688 = stablehlo.reshape %687 : (tensor<1xi64>) -> tensor<1x1xi64>
    %689 = stablehlo.transpose %arg158, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %690 = stablehlo.dot %632, %689, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %691 = stablehlo.reshape %690 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %692 = "stablehlo.scatter"(%arg232, %688, %691) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %693 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %694 = "stablehlo.gather"(%692, %693) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %695 = stablehlo.transpose %694, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %696 = stablehlo.dot_general %683, %695, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %697 = stablehlo.reshape %696 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %698 = stablehlo.transpose %arg157, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %699 = stablehlo.dot %697, %698, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %700 = stablehlo.add %620, %699 : tensor<1x4096xf32>
    %701 = stablehlo.power %700, %1 : tensor<1x4096xf32>
    %702 = stablehlo.reduce(%701 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %703 = stablehlo.multiply %702, %0 : tensor<1xf32>
    %704 = stablehlo.reshape %703 : (tensor<1xf32>) -> tensor<1x1xf32>
    %705 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %706 = stablehlo.add %704, %705 : tensor<1x1xf32>
    %707 = stablehlo.rsqrt %706 : tensor<1x1xf32>
    %708 = stablehlo.reshape %707 : (tensor<1x1xf32>) -> tensor<1xf32>
    %709 = stablehlo.broadcast_in_dim %708, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %710 = stablehlo.multiply %700, %709 : tensor<1x4096xf32>
    %711 = stablehlo.reshape %arg156 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %712 = stablehlo.multiply %710, %711 : tensor<1x4096xf32>
    %713 = stablehlo.transpose %arg236, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %714 = stablehlo.dot %712, %713, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %715 = stablehlo.logistic %714 : tensor<1x11008xf32>
    %716 = stablehlo.multiply %714, %715 : tensor<1x11008xf32>
    %717 = stablehlo.transpose %arg155, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %718 = stablehlo.dot %712, %717, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %719 = stablehlo.multiply %716, %718 : tensor<1x11008xf32>
    %720 = stablehlo.transpose %arg154, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %721 = stablehlo.dot %719, %720, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %722 = stablehlo.add %700, %721 : tensor<1x4096xf32>
    %723 = stablehlo.power %722, %1 : tensor<1x4096xf32>
    %724 = stablehlo.reduce(%723 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %725 = stablehlo.multiply %724, %0 : tensor<1xf32>
    %726 = stablehlo.reshape %725 : (tensor<1xf32>) -> tensor<1x1xf32>
    %727 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %728 = stablehlo.add %726, %727 : tensor<1x1xf32>
    %729 = stablehlo.rsqrt %728 : tensor<1x1xf32>
    %730 = stablehlo.reshape %729 : (tensor<1x1xf32>) -> tensor<1xf32>
    %731 = stablehlo.broadcast_in_dim %730, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %732 = stablehlo.multiply %722, %731 : tensor<1x4096xf32>
    %733 = stablehlo.reshape %arg153 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %734 = stablehlo.multiply %732, %733 : tensor<1x4096xf32>
    %735 = stablehlo.transpose %arg240, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %736 = stablehlo.dot %734, %735, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %737 = stablehlo.reshape %736 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %738 = stablehlo.slice %737 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %739 = stablehlo.reshape %738 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %740 = stablehlo.slice %737 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %741 = stablehlo.reshape %740 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %742 = stablehlo.complex %739, %741 : tensor<1x32x64xcomplex<f32>>
    %743 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %744 = stablehlo.multiply %742, %743 : tensor<1x32x64xcomplex<f32>>
    %745 = stablehlo.real %744 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %746 = stablehlo.reshape %745 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %747 = stablehlo.imag %744 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %748 = stablehlo.reshape %747 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %749 = stablehlo.concatenate %746, %748, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %750 = stablehlo.reshape %749 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %751 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %752 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %753 = stablehlo.add %arg198, %752 : tensor<1xi64>
    %754 = stablehlo.select %751, %753, %arg198 : tensor<1xi1>, tensor<1xi64>
    %755 = stablehlo.reshape %754 : (tensor<1xi64>) -> tensor<1x1xi64>
    %756 = stablehlo.transpose %arg238, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %757 = stablehlo.dot %734, %756, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %758 = stablehlo.reshape %757 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %759 = stablehlo.slice %758 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %760 = stablehlo.reshape %759 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %761 = stablehlo.slice %758 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %762 = stablehlo.reshape %761 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %763 = stablehlo.complex %760, %762 : tensor<1x32x64xcomplex<f32>>
    %764 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %765 = stablehlo.multiply %763, %764 : tensor<1x32x64xcomplex<f32>>
    %766 = stablehlo.real %765 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %767 = stablehlo.reshape %766 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %768 = stablehlo.imag %765 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %769 = stablehlo.reshape %768 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %770 = stablehlo.concatenate %767, %769, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %771 = stablehlo.reshape %770 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %772 = "stablehlo.scatter"(%arg239, %755, %771) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %773 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %774 = "stablehlo.gather"(%772, %773) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %775 = stablehlo.transpose %774, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %776 = stablehlo.dot_general %750, %775, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %777 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %778 = stablehlo.divide %776, %777 : tensor<32x1x2049xf32>
    %779 = stablehlo.reduce(%778 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %780 = stablehlo.broadcast_in_dim %779, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %781 = stablehlo.subtract %778, %780 : tensor<32x1x2049xf32>
    %782 = stablehlo.exponential %781 : tensor<32x1x2049xf32>
    %783 = stablehlo.reduce(%782 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %784 = stablehlo.broadcast_in_dim %783, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %785 = stablehlo.divide %782, %784 : tensor<32x1x2049xf32>
    %786 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %787 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %788 = stablehlo.add %arg198, %787 : tensor<1xi64>
    %789 = stablehlo.select %786, %788, %arg198 : tensor<1xi1>, tensor<1xi64>
    %790 = stablehlo.reshape %789 : (tensor<1xi64>) -> tensor<1x1xi64>
    %791 = stablehlo.transpose %arg152, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %792 = stablehlo.dot %734, %791, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %793 = stablehlo.reshape %792 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %794 = "stablehlo.scatter"(%arg237, %790, %793) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %795 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %796 = "stablehlo.gather"(%794, %795) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %797 = stablehlo.transpose %796, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %798 = stablehlo.dot_general %785, %797, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %799 = stablehlo.reshape %798 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %800 = stablehlo.transpose %arg151, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %801 = stablehlo.dot %799, %800, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %802 = stablehlo.add %722, %801 : tensor<1x4096xf32>
    %803 = stablehlo.power %802, %1 : tensor<1x4096xf32>
    %804 = stablehlo.reduce(%803 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %805 = stablehlo.multiply %804, %0 : tensor<1xf32>
    %806 = stablehlo.reshape %805 : (tensor<1xf32>) -> tensor<1x1xf32>
    %807 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %808 = stablehlo.add %806, %807 : tensor<1x1xf32>
    %809 = stablehlo.rsqrt %808 : tensor<1x1xf32>
    %810 = stablehlo.reshape %809 : (tensor<1x1xf32>) -> tensor<1xf32>
    %811 = stablehlo.broadcast_in_dim %810, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %812 = stablehlo.multiply %802, %811 : tensor<1x4096xf32>
    %813 = stablehlo.reshape %arg150 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %814 = stablehlo.multiply %812, %813 : tensor<1x4096xf32>
    %815 = stablehlo.transpose %arg241, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %816 = stablehlo.dot %814, %815, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %817 = stablehlo.logistic %816 : tensor<1x11008xf32>
    %818 = stablehlo.multiply %816, %817 : tensor<1x11008xf32>
    %819 = stablehlo.transpose %arg149, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %820 = stablehlo.dot %814, %819, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %821 = stablehlo.multiply %818, %820 : tensor<1x11008xf32>
    %822 = stablehlo.transpose %arg148, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %823 = stablehlo.dot %821, %822, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %824 = stablehlo.add %802, %823 : tensor<1x4096xf32>
    %825 = stablehlo.power %824, %1 : tensor<1x4096xf32>
    %826 = stablehlo.reduce(%825 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %827 = stablehlo.multiply %826, %0 : tensor<1xf32>
    %828 = stablehlo.reshape %827 : (tensor<1xf32>) -> tensor<1x1xf32>
    %829 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %830 = stablehlo.add %828, %829 : tensor<1x1xf32>
    %831 = stablehlo.rsqrt %830 : tensor<1x1xf32>
    %832 = stablehlo.reshape %831 : (tensor<1x1xf32>) -> tensor<1xf32>
    %833 = stablehlo.broadcast_in_dim %832, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %834 = stablehlo.multiply %824, %833 : tensor<1x4096xf32>
    %835 = stablehlo.reshape %arg147 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %836 = stablehlo.multiply %834, %835 : tensor<1x4096xf32>
    %837 = stablehlo.transpose %arg245, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %838 = stablehlo.dot %836, %837, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %839 = stablehlo.reshape %838 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %840 = stablehlo.slice %839 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %841 = stablehlo.reshape %840 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %842 = stablehlo.slice %839 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %843 = stablehlo.reshape %842 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %844 = stablehlo.complex %841, %843 : tensor<1x32x64xcomplex<f32>>
    %845 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %846 = stablehlo.multiply %844, %845 : tensor<1x32x64xcomplex<f32>>
    %847 = stablehlo.real %846 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %848 = stablehlo.reshape %847 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %849 = stablehlo.imag %846 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %850 = stablehlo.reshape %849 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %851 = stablehlo.concatenate %848, %850, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %852 = stablehlo.reshape %851 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %853 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %854 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %855 = stablehlo.add %arg198, %854 : tensor<1xi64>
    %856 = stablehlo.select %853, %855, %arg198 : tensor<1xi1>, tensor<1xi64>
    %857 = stablehlo.reshape %856 : (tensor<1xi64>) -> tensor<1x1xi64>
    %858 = stablehlo.transpose %arg243, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %859 = stablehlo.dot %836, %858, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %860 = stablehlo.reshape %859 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %861 = stablehlo.slice %860 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %862 = stablehlo.reshape %861 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %863 = stablehlo.slice %860 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %864 = stablehlo.reshape %863 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %865 = stablehlo.complex %862, %864 : tensor<1x32x64xcomplex<f32>>
    %866 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %867 = stablehlo.multiply %865, %866 : tensor<1x32x64xcomplex<f32>>
    %868 = stablehlo.real %867 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %869 = stablehlo.reshape %868 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %870 = stablehlo.imag %867 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %871 = stablehlo.reshape %870 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %872 = stablehlo.concatenate %869, %871, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %873 = stablehlo.reshape %872 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %874 = "stablehlo.scatter"(%arg244, %857, %873) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %875 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %876 = "stablehlo.gather"(%874, %875) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %877 = stablehlo.transpose %876, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %878 = stablehlo.dot_general %852, %877, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %879 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %880 = stablehlo.divide %878, %879 : tensor<32x1x2049xf32>
    %881 = stablehlo.reduce(%880 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %882 = stablehlo.broadcast_in_dim %881, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %883 = stablehlo.subtract %880, %882 : tensor<32x1x2049xf32>
    %884 = stablehlo.exponential %883 : tensor<32x1x2049xf32>
    %885 = stablehlo.reduce(%884 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %886 = stablehlo.broadcast_in_dim %885, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %887 = stablehlo.divide %884, %886 : tensor<32x1x2049xf32>
    %888 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %889 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %890 = stablehlo.add %arg198, %889 : tensor<1xi64>
    %891 = stablehlo.select %888, %890, %arg198 : tensor<1xi1>, tensor<1xi64>
    %892 = stablehlo.reshape %891 : (tensor<1xi64>) -> tensor<1x1xi64>
    %893 = stablehlo.transpose %arg146, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %894 = stablehlo.dot %836, %893, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %895 = stablehlo.reshape %894 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %896 = "stablehlo.scatter"(%arg242, %892, %895) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %897 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %898 = "stablehlo.gather"(%896, %897) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %899 = stablehlo.transpose %898, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %900 = stablehlo.dot_general %887, %899, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %901 = stablehlo.reshape %900 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %902 = stablehlo.transpose %arg145, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %903 = stablehlo.dot %901, %902, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %904 = stablehlo.add %824, %903 : tensor<1x4096xf32>
    %905 = stablehlo.power %904, %1 : tensor<1x4096xf32>
    %906 = stablehlo.reduce(%905 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %907 = stablehlo.multiply %906, %0 : tensor<1xf32>
    %908 = stablehlo.reshape %907 : (tensor<1xf32>) -> tensor<1x1xf32>
    %909 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %910 = stablehlo.add %908, %909 : tensor<1x1xf32>
    %911 = stablehlo.rsqrt %910 : tensor<1x1xf32>
    %912 = stablehlo.reshape %911 : (tensor<1x1xf32>) -> tensor<1xf32>
    %913 = stablehlo.broadcast_in_dim %912, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %914 = stablehlo.multiply %904, %913 : tensor<1x4096xf32>
    %915 = stablehlo.reshape %arg144 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %916 = stablehlo.multiply %914, %915 : tensor<1x4096xf32>
    %917 = stablehlo.transpose %arg246, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %918 = stablehlo.dot %916, %917, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %919 = stablehlo.logistic %918 : tensor<1x11008xf32>
    %920 = stablehlo.multiply %918, %919 : tensor<1x11008xf32>
    %921 = stablehlo.transpose %arg143, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %922 = stablehlo.dot %916, %921, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %923 = stablehlo.multiply %920, %922 : tensor<1x11008xf32>
    %924 = stablehlo.transpose %arg142, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %925 = stablehlo.dot %923, %924, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %926 = stablehlo.add %904, %925 : tensor<1x4096xf32>
    %927 = stablehlo.power %926, %1 : tensor<1x4096xf32>
    %928 = stablehlo.reduce(%927 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %929 = stablehlo.multiply %928, %0 : tensor<1xf32>
    %930 = stablehlo.reshape %929 : (tensor<1xf32>) -> tensor<1x1xf32>
    %931 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %932 = stablehlo.add %930, %931 : tensor<1x1xf32>
    %933 = stablehlo.rsqrt %932 : tensor<1x1xf32>
    %934 = stablehlo.reshape %933 : (tensor<1x1xf32>) -> tensor<1xf32>
    %935 = stablehlo.broadcast_in_dim %934, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %936 = stablehlo.multiply %926, %935 : tensor<1x4096xf32>
    %937 = stablehlo.reshape %arg141 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %938 = stablehlo.multiply %936, %937 : tensor<1x4096xf32>
    %939 = stablehlo.transpose %arg250, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %940 = stablehlo.dot %938, %939, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %941 = stablehlo.reshape %940 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %942 = stablehlo.slice %941 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %943 = stablehlo.reshape %942 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %944 = stablehlo.slice %941 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %945 = stablehlo.reshape %944 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %946 = stablehlo.complex %943, %945 : tensor<1x32x64xcomplex<f32>>
    %947 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %948 = stablehlo.multiply %946, %947 : tensor<1x32x64xcomplex<f32>>
    %949 = stablehlo.real %948 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %950 = stablehlo.reshape %949 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %951 = stablehlo.imag %948 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %952 = stablehlo.reshape %951 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %953 = stablehlo.concatenate %950, %952, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %954 = stablehlo.reshape %953 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %955 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %956 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %957 = stablehlo.add %arg198, %956 : tensor<1xi64>
    %958 = stablehlo.select %955, %957, %arg198 : tensor<1xi1>, tensor<1xi64>
    %959 = stablehlo.reshape %958 : (tensor<1xi64>) -> tensor<1x1xi64>
    %960 = stablehlo.transpose %arg248, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %961 = stablehlo.dot %938, %960, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %962 = stablehlo.reshape %961 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %963 = stablehlo.slice %962 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %964 = stablehlo.reshape %963 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %965 = stablehlo.slice %962 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %966 = stablehlo.reshape %965 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %967 = stablehlo.complex %964, %966 : tensor<1x32x64xcomplex<f32>>
    %968 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %969 = stablehlo.multiply %967, %968 : tensor<1x32x64xcomplex<f32>>
    %970 = stablehlo.real %969 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %971 = stablehlo.reshape %970 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %972 = stablehlo.imag %969 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %973 = stablehlo.reshape %972 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %974 = stablehlo.concatenate %971, %973, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %975 = stablehlo.reshape %974 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %976 = "stablehlo.scatter"(%arg249, %959, %975) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %977 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %978 = "stablehlo.gather"(%976, %977) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %979 = stablehlo.transpose %978, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %980 = stablehlo.dot_general %954, %979, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %981 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %982 = stablehlo.divide %980, %981 : tensor<32x1x2049xf32>
    %983 = stablehlo.reduce(%982 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %984 = stablehlo.broadcast_in_dim %983, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %985 = stablehlo.subtract %982, %984 : tensor<32x1x2049xf32>
    %986 = stablehlo.exponential %985 : tensor<32x1x2049xf32>
    %987 = stablehlo.reduce(%986 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %988 = stablehlo.broadcast_in_dim %987, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %989 = stablehlo.divide %986, %988 : tensor<32x1x2049xf32>
    %990 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %991 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %992 = stablehlo.add %arg198, %991 : tensor<1xi64>
    %993 = stablehlo.select %990, %992, %arg198 : tensor<1xi1>, tensor<1xi64>
    %994 = stablehlo.reshape %993 : (tensor<1xi64>) -> tensor<1x1xi64>
    %995 = stablehlo.transpose %arg140, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %996 = stablehlo.dot %938, %995, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %997 = stablehlo.reshape %996 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %998 = "stablehlo.scatter"(%arg247, %994, %997) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %999 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1000 = "stablehlo.gather"(%998, %999) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1001 = stablehlo.transpose %1000, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %1002 = stablehlo.dot_general %989, %1001, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %1003 = stablehlo.reshape %1002 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %1004 = stablehlo.transpose %arg139, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1005 = stablehlo.dot %1003, %1004, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1006 = stablehlo.add %926, %1005 : tensor<1x4096xf32>
    %1007 = stablehlo.power %1006, %1 : tensor<1x4096xf32>
    %1008 = stablehlo.reduce(%1007 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1009 = stablehlo.multiply %1008, %0 : tensor<1xf32>
    %1010 = stablehlo.reshape %1009 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1011 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1012 = stablehlo.add %1010, %1011 : tensor<1x1xf32>
    %1013 = stablehlo.rsqrt %1012 : tensor<1x1xf32>
    %1014 = stablehlo.reshape %1013 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1015 = stablehlo.broadcast_in_dim %1014, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1016 = stablehlo.multiply %1006, %1015 : tensor<1x4096xf32>
    %1017 = stablehlo.reshape %arg138 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1018 = stablehlo.multiply %1016, %1017 : tensor<1x4096xf32>
    %1019 = stablehlo.transpose %arg251, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1020 = stablehlo.dot %1018, %1019, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1021 = stablehlo.logistic %1020 : tensor<1x11008xf32>
    %1022 = stablehlo.multiply %1020, %1021 : tensor<1x11008xf32>
    %1023 = stablehlo.transpose %arg137, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1024 = stablehlo.dot %1018, %1023, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1025 = stablehlo.multiply %1022, %1024 : tensor<1x11008xf32>
    %1026 = stablehlo.transpose %arg136, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1027 = stablehlo.dot %1025, %1026, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %1028 = stablehlo.add %1006, %1027 : tensor<1x4096xf32>
    %1029 = stablehlo.power %1028, %1 : tensor<1x4096xf32>
    %1030 = stablehlo.reduce(%1029 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1031 = stablehlo.multiply %1030, %0 : tensor<1xf32>
    %1032 = stablehlo.reshape %1031 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1033 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1034 = stablehlo.add %1032, %1033 : tensor<1x1xf32>
    %1035 = stablehlo.rsqrt %1034 : tensor<1x1xf32>
    %1036 = stablehlo.reshape %1035 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1037 = stablehlo.broadcast_in_dim %1036, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1038 = stablehlo.multiply %1028, %1037 : tensor<1x4096xf32>
    %1039 = stablehlo.reshape %arg135 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1040 = stablehlo.multiply %1038, %1039 : tensor<1x4096xf32>
    %1041 = stablehlo.transpose %arg255, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1042 = stablehlo.dot %1040, %1041, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1043 = stablehlo.reshape %1042 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1044 = stablehlo.slice %1043 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1045 = stablehlo.reshape %1044 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1046 = stablehlo.slice %1043 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1047 = stablehlo.reshape %1046 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1048 = stablehlo.complex %1045, %1047 : tensor<1x32x64xcomplex<f32>>
    %1049 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1050 = stablehlo.multiply %1048, %1049 : tensor<1x32x64xcomplex<f32>>
    %1051 = stablehlo.real %1050 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1052 = stablehlo.reshape %1051 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1053 = stablehlo.imag %1050 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1054 = stablehlo.reshape %1053 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1055 = stablehlo.concatenate %1052, %1054, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1056 = stablehlo.reshape %1055 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %1057 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1058 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1059 = stablehlo.add %arg198, %1058 : tensor<1xi64>
    %1060 = stablehlo.select %1057, %1059, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1061 = stablehlo.reshape %1060 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1062 = stablehlo.transpose %arg253, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1063 = stablehlo.dot %1040, %1062, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1064 = stablehlo.reshape %1063 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1065 = stablehlo.slice %1064 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1066 = stablehlo.reshape %1065 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1067 = stablehlo.slice %1064 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1068 = stablehlo.reshape %1067 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1069 = stablehlo.complex %1066, %1068 : tensor<1x32x64xcomplex<f32>>
    %1070 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1071 = stablehlo.multiply %1069, %1070 : tensor<1x32x64xcomplex<f32>>
    %1072 = stablehlo.real %1071 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1073 = stablehlo.reshape %1072 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1074 = stablehlo.imag %1071 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1075 = stablehlo.reshape %1074 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1076 = stablehlo.concatenate %1073, %1075, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1077 = stablehlo.reshape %1076 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %1078 = "stablehlo.scatter"(%arg254, %1061, %1077) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1079 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1080 = "stablehlo.gather"(%1078, %1079) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1081 = stablehlo.transpose %1080, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %1082 = stablehlo.dot_general %1056, %1081, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %1083 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %1084 = stablehlo.divide %1082, %1083 : tensor<32x1x2049xf32>
    %1085 = stablehlo.reduce(%1084 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1086 = stablehlo.broadcast_in_dim %1085, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1087 = stablehlo.subtract %1084, %1086 : tensor<32x1x2049xf32>
    %1088 = stablehlo.exponential %1087 : tensor<32x1x2049xf32>
    %1089 = stablehlo.reduce(%1088 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1090 = stablehlo.broadcast_in_dim %1089, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1091 = stablehlo.divide %1088, %1090 : tensor<32x1x2049xf32>
    %1092 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1093 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1094 = stablehlo.add %arg198, %1093 : tensor<1xi64>
    %1095 = stablehlo.select %1092, %1094, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1096 = stablehlo.reshape %1095 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1097 = stablehlo.transpose %arg134, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1098 = stablehlo.dot %1040, %1097, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1099 = stablehlo.reshape %1098 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %1100 = "stablehlo.scatter"(%arg252, %1096, %1099) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1101 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1102 = "stablehlo.gather"(%1100, %1101) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1103 = stablehlo.transpose %1102, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %1104 = stablehlo.dot_general %1091, %1103, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %1105 = stablehlo.reshape %1104 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %1106 = stablehlo.transpose %arg133, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1107 = stablehlo.dot %1105, %1106, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1108 = stablehlo.add %1028, %1107 : tensor<1x4096xf32>
    %1109 = stablehlo.power %1108, %1 : tensor<1x4096xf32>
    %1110 = stablehlo.reduce(%1109 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1111 = stablehlo.multiply %1110, %0 : tensor<1xf32>
    %1112 = stablehlo.reshape %1111 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1113 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1114 = stablehlo.add %1112, %1113 : tensor<1x1xf32>
    %1115 = stablehlo.rsqrt %1114 : tensor<1x1xf32>
    %1116 = stablehlo.reshape %1115 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1117 = stablehlo.broadcast_in_dim %1116, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1118 = stablehlo.multiply %1108, %1117 : tensor<1x4096xf32>
    %1119 = stablehlo.reshape %arg132 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1120 = stablehlo.multiply %1118, %1119 : tensor<1x4096xf32>
    %1121 = stablehlo.transpose %arg256, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1122 = stablehlo.dot %1120, %1121, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1123 = stablehlo.logistic %1122 : tensor<1x11008xf32>
    %1124 = stablehlo.multiply %1122, %1123 : tensor<1x11008xf32>
    %1125 = stablehlo.transpose %arg131, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1126 = stablehlo.dot %1120, %1125, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1127 = stablehlo.multiply %1124, %1126 : tensor<1x11008xf32>
    %1128 = stablehlo.transpose %arg130, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1129 = stablehlo.dot %1127, %1128, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %1130 = stablehlo.add %1108, %1129 : tensor<1x4096xf32>
    %1131 = stablehlo.power %1130, %1 : tensor<1x4096xf32>
    %1132 = stablehlo.reduce(%1131 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1133 = stablehlo.multiply %1132, %0 : tensor<1xf32>
    %1134 = stablehlo.reshape %1133 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1135 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1136 = stablehlo.add %1134, %1135 : tensor<1x1xf32>
    %1137 = stablehlo.rsqrt %1136 : tensor<1x1xf32>
    %1138 = stablehlo.reshape %1137 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1139 = stablehlo.broadcast_in_dim %1138, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1140 = stablehlo.multiply %1130, %1139 : tensor<1x4096xf32>
    %1141 = stablehlo.reshape %arg129 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1142 = stablehlo.multiply %1140, %1141 : tensor<1x4096xf32>
    %1143 = stablehlo.transpose %arg260, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1144 = stablehlo.dot %1142, %1143, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1145 = stablehlo.reshape %1144 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1146 = stablehlo.slice %1145 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1147 = stablehlo.reshape %1146 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1148 = stablehlo.slice %1145 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1149 = stablehlo.reshape %1148 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1150 = stablehlo.complex %1147, %1149 : tensor<1x32x64xcomplex<f32>>
    %1151 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1152 = stablehlo.multiply %1150, %1151 : tensor<1x32x64xcomplex<f32>>
    %1153 = stablehlo.real %1152 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1154 = stablehlo.reshape %1153 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1155 = stablehlo.imag %1152 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1156 = stablehlo.reshape %1155 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1157 = stablehlo.concatenate %1154, %1156, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1158 = stablehlo.reshape %1157 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %1159 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1160 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1161 = stablehlo.add %arg198, %1160 : tensor<1xi64>
    %1162 = stablehlo.select %1159, %1161, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1163 = stablehlo.reshape %1162 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1164 = stablehlo.transpose %arg258, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1165 = stablehlo.dot %1142, %1164, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1166 = stablehlo.reshape %1165 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1167 = stablehlo.slice %1166 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1168 = stablehlo.reshape %1167 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1169 = stablehlo.slice %1166 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1170 = stablehlo.reshape %1169 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1171 = stablehlo.complex %1168, %1170 : tensor<1x32x64xcomplex<f32>>
    %1172 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1173 = stablehlo.multiply %1171, %1172 : tensor<1x32x64xcomplex<f32>>
    %1174 = stablehlo.real %1173 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1175 = stablehlo.reshape %1174 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1176 = stablehlo.imag %1173 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1177 = stablehlo.reshape %1176 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1178 = stablehlo.concatenate %1175, %1177, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1179 = stablehlo.reshape %1178 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %1180 = "stablehlo.scatter"(%arg259, %1163, %1179) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1181 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1182 = "stablehlo.gather"(%1180, %1181) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1183 = stablehlo.transpose %1182, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %1184 = stablehlo.dot_general %1158, %1183, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %1185 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %1186 = stablehlo.divide %1184, %1185 : tensor<32x1x2049xf32>
    %1187 = stablehlo.reduce(%1186 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1188 = stablehlo.broadcast_in_dim %1187, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1189 = stablehlo.subtract %1186, %1188 : tensor<32x1x2049xf32>
    %1190 = stablehlo.exponential %1189 : tensor<32x1x2049xf32>
    %1191 = stablehlo.reduce(%1190 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1192 = stablehlo.broadcast_in_dim %1191, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1193 = stablehlo.divide %1190, %1192 : tensor<32x1x2049xf32>
    %1194 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1195 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1196 = stablehlo.add %arg198, %1195 : tensor<1xi64>
    %1197 = stablehlo.select %1194, %1196, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1198 = stablehlo.reshape %1197 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1199 = stablehlo.transpose %arg128, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1200 = stablehlo.dot %1142, %1199, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1201 = stablehlo.reshape %1200 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %1202 = "stablehlo.scatter"(%arg257, %1198, %1201) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1203 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1204 = "stablehlo.gather"(%1202, %1203) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1205 = stablehlo.transpose %1204, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %1206 = stablehlo.dot_general %1193, %1205, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %1207 = stablehlo.reshape %1206 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %1208 = stablehlo.transpose %arg127, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1209 = stablehlo.dot %1207, %1208, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1210 = stablehlo.add %1130, %1209 : tensor<1x4096xf32>
    %1211 = stablehlo.power %1210, %1 : tensor<1x4096xf32>
    %1212 = stablehlo.reduce(%1211 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1213 = stablehlo.multiply %1212, %0 : tensor<1xf32>
    %1214 = stablehlo.reshape %1213 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1215 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1216 = stablehlo.add %1214, %1215 : tensor<1x1xf32>
    %1217 = stablehlo.rsqrt %1216 : tensor<1x1xf32>
    %1218 = stablehlo.reshape %1217 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1219 = stablehlo.broadcast_in_dim %1218, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1220 = stablehlo.multiply %1210, %1219 : tensor<1x4096xf32>
    %1221 = stablehlo.reshape %arg126 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1222 = stablehlo.multiply %1220, %1221 : tensor<1x4096xf32>
    %1223 = stablehlo.transpose %arg261, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1224 = stablehlo.dot %1222, %1223, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1225 = stablehlo.logistic %1224 : tensor<1x11008xf32>
    %1226 = stablehlo.multiply %1224, %1225 : tensor<1x11008xf32>
    %1227 = stablehlo.transpose %arg125, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1228 = stablehlo.dot %1222, %1227, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1229 = stablehlo.multiply %1226, %1228 : tensor<1x11008xf32>
    %1230 = stablehlo.transpose %arg124, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1231 = stablehlo.dot %1229, %1230, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %1232 = stablehlo.add %1210, %1231 : tensor<1x4096xf32>
    %1233 = stablehlo.power %1232, %1 : tensor<1x4096xf32>
    %1234 = stablehlo.reduce(%1233 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1235 = stablehlo.multiply %1234, %0 : tensor<1xf32>
    %1236 = stablehlo.reshape %1235 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1237 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1238 = stablehlo.add %1236, %1237 : tensor<1x1xf32>
    %1239 = stablehlo.rsqrt %1238 : tensor<1x1xf32>
    %1240 = stablehlo.reshape %1239 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1241 = stablehlo.broadcast_in_dim %1240, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1242 = stablehlo.multiply %1232, %1241 : tensor<1x4096xf32>
    %1243 = stablehlo.reshape %arg123 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1244 = stablehlo.multiply %1242, %1243 : tensor<1x4096xf32>
    %1245 = stablehlo.transpose %arg265, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1246 = stablehlo.dot %1244, %1245, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1247 = stablehlo.reshape %1246 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1248 = stablehlo.slice %1247 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1249 = stablehlo.reshape %1248 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1250 = stablehlo.slice %1247 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1251 = stablehlo.reshape %1250 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1252 = stablehlo.complex %1249, %1251 : tensor<1x32x64xcomplex<f32>>
    %1253 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1254 = stablehlo.multiply %1252, %1253 : tensor<1x32x64xcomplex<f32>>
    %1255 = stablehlo.real %1254 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1256 = stablehlo.reshape %1255 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1257 = stablehlo.imag %1254 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1258 = stablehlo.reshape %1257 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1259 = stablehlo.concatenate %1256, %1258, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1260 = stablehlo.reshape %1259 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %1261 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1262 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1263 = stablehlo.add %arg198, %1262 : tensor<1xi64>
    %1264 = stablehlo.select %1261, %1263, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1265 = stablehlo.reshape %1264 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1266 = stablehlo.transpose %arg263, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1267 = stablehlo.dot %1244, %1266, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1268 = stablehlo.reshape %1267 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1269 = stablehlo.slice %1268 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1270 = stablehlo.reshape %1269 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1271 = stablehlo.slice %1268 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1272 = stablehlo.reshape %1271 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1273 = stablehlo.complex %1270, %1272 : tensor<1x32x64xcomplex<f32>>
    %1274 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1275 = stablehlo.multiply %1273, %1274 : tensor<1x32x64xcomplex<f32>>
    %1276 = stablehlo.real %1275 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1277 = stablehlo.reshape %1276 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1278 = stablehlo.imag %1275 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1279 = stablehlo.reshape %1278 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1280 = stablehlo.concatenate %1277, %1279, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1281 = stablehlo.reshape %1280 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %1282 = "stablehlo.scatter"(%arg264, %1265, %1281) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1283 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1284 = "stablehlo.gather"(%1282, %1283) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1285 = stablehlo.transpose %1284, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %1286 = stablehlo.dot_general %1260, %1285, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %1287 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %1288 = stablehlo.divide %1286, %1287 : tensor<32x1x2049xf32>
    %1289 = stablehlo.reduce(%1288 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1290 = stablehlo.broadcast_in_dim %1289, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1291 = stablehlo.subtract %1288, %1290 : tensor<32x1x2049xf32>
    %1292 = stablehlo.exponential %1291 : tensor<32x1x2049xf32>
    %1293 = stablehlo.reduce(%1292 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1294 = stablehlo.broadcast_in_dim %1293, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1295 = stablehlo.divide %1292, %1294 : tensor<32x1x2049xf32>
    %1296 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1297 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1298 = stablehlo.add %arg198, %1297 : tensor<1xi64>
    %1299 = stablehlo.select %1296, %1298, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1300 = stablehlo.reshape %1299 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1301 = stablehlo.transpose %arg122, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1302 = stablehlo.dot %1244, %1301, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1303 = stablehlo.reshape %1302 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %1304 = "stablehlo.scatter"(%arg262, %1300, %1303) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1305 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1306 = "stablehlo.gather"(%1304, %1305) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1307 = stablehlo.transpose %1306, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %1308 = stablehlo.dot_general %1295, %1307, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %1309 = stablehlo.reshape %1308 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %1310 = stablehlo.transpose %arg121, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1311 = stablehlo.dot %1309, %1310, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1312 = stablehlo.add %1232, %1311 : tensor<1x4096xf32>
    %1313 = stablehlo.power %1312, %1 : tensor<1x4096xf32>
    %1314 = stablehlo.reduce(%1313 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1315 = stablehlo.multiply %1314, %0 : tensor<1xf32>
    %1316 = stablehlo.reshape %1315 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1317 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1318 = stablehlo.add %1316, %1317 : tensor<1x1xf32>
    %1319 = stablehlo.rsqrt %1318 : tensor<1x1xf32>
    %1320 = stablehlo.reshape %1319 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1321 = stablehlo.broadcast_in_dim %1320, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1322 = stablehlo.multiply %1312, %1321 : tensor<1x4096xf32>
    %1323 = stablehlo.reshape %arg120 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1324 = stablehlo.multiply %1322, %1323 : tensor<1x4096xf32>
    %1325 = stablehlo.transpose %arg266, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1326 = stablehlo.dot %1324, %1325, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1327 = stablehlo.logistic %1326 : tensor<1x11008xf32>
    %1328 = stablehlo.multiply %1326, %1327 : tensor<1x11008xf32>
    %1329 = stablehlo.transpose %arg119, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1330 = stablehlo.dot %1324, %1329, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1331 = stablehlo.multiply %1328, %1330 : tensor<1x11008xf32>
    %1332 = stablehlo.transpose %arg118, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1333 = stablehlo.dot %1331, %1332, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %1334 = stablehlo.add %1312, %1333 : tensor<1x4096xf32>
    %1335 = stablehlo.power %1334, %1 : tensor<1x4096xf32>
    %1336 = stablehlo.reduce(%1335 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1337 = stablehlo.multiply %1336, %0 : tensor<1xf32>
    %1338 = stablehlo.reshape %1337 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1339 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1340 = stablehlo.add %1338, %1339 : tensor<1x1xf32>
    %1341 = stablehlo.rsqrt %1340 : tensor<1x1xf32>
    %1342 = stablehlo.reshape %1341 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1343 = stablehlo.broadcast_in_dim %1342, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1344 = stablehlo.multiply %1334, %1343 : tensor<1x4096xf32>
    %1345 = stablehlo.reshape %arg117 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1346 = stablehlo.multiply %1344, %1345 : tensor<1x4096xf32>
    %1347 = stablehlo.transpose %arg270, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1348 = stablehlo.dot %1346, %1347, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1349 = stablehlo.reshape %1348 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1350 = stablehlo.slice %1349 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1351 = stablehlo.reshape %1350 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1352 = stablehlo.slice %1349 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1353 = stablehlo.reshape %1352 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1354 = stablehlo.complex %1351, %1353 : tensor<1x32x64xcomplex<f32>>
    %1355 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1356 = stablehlo.multiply %1354, %1355 : tensor<1x32x64xcomplex<f32>>
    %1357 = stablehlo.real %1356 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1358 = stablehlo.reshape %1357 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1359 = stablehlo.imag %1356 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1360 = stablehlo.reshape %1359 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1361 = stablehlo.concatenate %1358, %1360, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1362 = stablehlo.reshape %1361 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %1363 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1364 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1365 = stablehlo.add %arg198, %1364 : tensor<1xi64>
    %1366 = stablehlo.select %1363, %1365, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1367 = stablehlo.reshape %1366 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1368 = stablehlo.transpose %arg268, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1369 = stablehlo.dot %1346, %1368, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1370 = stablehlo.reshape %1369 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1371 = stablehlo.slice %1370 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1372 = stablehlo.reshape %1371 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1373 = stablehlo.slice %1370 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1374 = stablehlo.reshape %1373 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1375 = stablehlo.complex %1372, %1374 : tensor<1x32x64xcomplex<f32>>
    %1376 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1377 = stablehlo.multiply %1375, %1376 : tensor<1x32x64xcomplex<f32>>
    %1378 = stablehlo.real %1377 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1379 = stablehlo.reshape %1378 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1380 = stablehlo.imag %1377 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1381 = stablehlo.reshape %1380 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1382 = stablehlo.concatenate %1379, %1381, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1383 = stablehlo.reshape %1382 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %1384 = "stablehlo.scatter"(%arg269, %1367, %1383) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1385 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1386 = "stablehlo.gather"(%1384, %1385) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1387 = stablehlo.transpose %1386, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %1388 = stablehlo.dot_general %1362, %1387, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %1389 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %1390 = stablehlo.divide %1388, %1389 : tensor<32x1x2049xf32>
    %1391 = stablehlo.reduce(%1390 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1392 = stablehlo.broadcast_in_dim %1391, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1393 = stablehlo.subtract %1390, %1392 : tensor<32x1x2049xf32>
    %1394 = stablehlo.exponential %1393 : tensor<32x1x2049xf32>
    %1395 = stablehlo.reduce(%1394 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1396 = stablehlo.broadcast_in_dim %1395, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1397 = stablehlo.divide %1394, %1396 : tensor<32x1x2049xf32>
    %1398 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1399 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1400 = stablehlo.add %arg198, %1399 : tensor<1xi64>
    %1401 = stablehlo.select %1398, %1400, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1402 = stablehlo.reshape %1401 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1403 = stablehlo.transpose %arg116, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1404 = stablehlo.dot %1346, %1403, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1405 = stablehlo.reshape %1404 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %1406 = "stablehlo.scatter"(%arg267, %1402, %1405) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1407 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1408 = "stablehlo.gather"(%1406, %1407) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1409 = stablehlo.transpose %1408, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %1410 = stablehlo.dot_general %1397, %1409, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %1411 = stablehlo.reshape %1410 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %1412 = stablehlo.transpose %arg115, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1413 = stablehlo.dot %1411, %1412, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1414 = stablehlo.add %1334, %1413 : tensor<1x4096xf32>
    %1415 = stablehlo.power %1414, %1 : tensor<1x4096xf32>
    %1416 = stablehlo.reduce(%1415 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1417 = stablehlo.multiply %1416, %0 : tensor<1xf32>
    %1418 = stablehlo.reshape %1417 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1419 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1420 = stablehlo.add %1418, %1419 : tensor<1x1xf32>
    %1421 = stablehlo.rsqrt %1420 : tensor<1x1xf32>
    %1422 = stablehlo.reshape %1421 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1423 = stablehlo.broadcast_in_dim %1422, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1424 = stablehlo.multiply %1414, %1423 : tensor<1x4096xf32>
    %1425 = stablehlo.reshape %arg114 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1426 = stablehlo.multiply %1424, %1425 : tensor<1x4096xf32>
    %1427 = stablehlo.transpose %arg271, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1428 = stablehlo.dot %1426, %1427, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1429 = stablehlo.logistic %1428 : tensor<1x11008xf32>
    %1430 = stablehlo.multiply %1428, %1429 : tensor<1x11008xf32>
    %1431 = stablehlo.transpose %arg113, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1432 = stablehlo.dot %1426, %1431, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1433 = stablehlo.multiply %1430, %1432 : tensor<1x11008xf32>
    %1434 = stablehlo.transpose %arg112, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1435 = stablehlo.dot %1433, %1434, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %1436 = stablehlo.add %1414, %1435 : tensor<1x4096xf32>
    %1437 = stablehlo.power %1436, %1 : tensor<1x4096xf32>
    %1438 = stablehlo.reduce(%1437 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1439 = stablehlo.multiply %1438, %0 : tensor<1xf32>
    %1440 = stablehlo.reshape %1439 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1441 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1442 = stablehlo.add %1440, %1441 : tensor<1x1xf32>
    %1443 = stablehlo.rsqrt %1442 : tensor<1x1xf32>
    %1444 = stablehlo.reshape %1443 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1445 = stablehlo.broadcast_in_dim %1444, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1446 = stablehlo.multiply %1436, %1445 : tensor<1x4096xf32>
    %1447 = stablehlo.reshape %arg111 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1448 = stablehlo.multiply %1446, %1447 : tensor<1x4096xf32>
    %1449 = stablehlo.transpose %arg275, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1450 = stablehlo.dot %1448, %1449, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1451 = stablehlo.reshape %1450 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1452 = stablehlo.slice %1451 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1453 = stablehlo.reshape %1452 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1454 = stablehlo.slice %1451 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1455 = stablehlo.reshape %1454 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1456 = stablehlo.complex %1453, %1455 : tensor<1x32x64xcomplex<f32>>
    %1457 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1458 = stablehlo.multiply %1456, %1457 : tensor<1x32x64xcomplex<f32>>
    %1459 = stablehlo.real %1458 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1460 = stablehlo.reshape %1459 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1461 = stablehlo.imag %1458 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1462 = stablehlo.reshape %1461 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1463 = stablehlo.concatenate %1460, %1462, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1464 = stablehlo.reshape %1463 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %1465 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1466 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1467 = stablehlo.add %arg198, %1466 : tensor<1xi64>
    %1468 = stablehlo.select %1465, %1467, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1469 = stablehlo.reshape %1468 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1470 = stablehlo.transpose %arg273, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1471 = stablehlo.dot %1448, %1470, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1472 = stablehlo.reshape %1471 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1473 = stablehlo.slice %1472 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1474 = stablehlo.reshape %1473 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1475 = stablehlo.slice %1472 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1476 = stablehlo.reshape %1475 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1477 = stablehlo.complex %1474, %1476 : tensor<1x32x64xcomplex<f32>>
    %1478 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1479 = stablehlo.multiply %1477, %1478 : tensor<1x32x64xcomplex<f32>>
    %1480 = stablehlo.real %1479 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1481 = stablehlo.reshape %1480 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1482 = stablehlo.imag %1479 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1483 = stablehlo.reshape %1482 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1484 = stablehlo.concatenate %1481, %1483, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1485 = stablehlo.reshape %1484 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %1486 = "stablehlo.scatter"(%arg274, %1469, %1485) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1487 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1488 = "stablehlo.gather"(%1486, %1487) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1489 = stablehlo.transpose %1488, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %1490 = stablehlo.dot_general %1464, %1489, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %1491 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %1492 = stablehlo.divide %1490, %1491 : tensor<32x1x2049xf32>
    %1493 = stablehlo.reduce(%1492 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1494 = stablehlo.broadcast_in_dim %1493, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1495 = stablehlo.subtract %1492, %1494 : tensor<32x1x2049xf32>
    %1496 = stablehlo.exponential %1495 : tensor<32x1x2049xf32>
    %1497 = stablehlo.reduce(%1496 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1498 = stablehlo.broadcast_in_dim %1497, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1499 = stablehlo.divide %1496, %1498 : tensor<32x1x2049xf32>
    %1500 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1501 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1502 = stablehlo.add %arg198, %1501 : tensor<1xi64>
    %1503 = stablehlo.select %1500, %1502, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1504 = stablehlo.reshape %1503 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1505 = stablehlo.transpose %arg110, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1506 = stablehlo.dot %1448, %1505, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1507 = stablehlo.reshape %1506 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %1508 = "stablehlo.scatter"(%arg272, %1504, %1507) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1509 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1510 = "stablehlo.gather"(%1508, %1509) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1511 = stablehlo.transpose %1510, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %1512 = stablehlo.dot_general %1499, %1511, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %1513 = stablehlo.reshape %1512 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %1514 = stablehlo.transpose %arg109, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1515 = stablehlo.dot %1513, %1514, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1516 = stablehlo.add %1436, %1515 : tensor<1x4096xf32>
    %1517 = stablehlo.power %1516, %1 : tensor<1x4096xf32>
    %1518 = stablehlo.reduce(%1517 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1519 = stablehlo.multiply %1518, %0 : tensor<1xf32>
    %1520 = stablehlo.reshape %1519 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1521 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1522 = stablehlo.add %1520, %1521 : tensor<1x1xf32>
    %1523 = stablehlo.rsqrt %1522 : tensor<1x1xf32>
    %1524 = stablehlo.reshape %1523 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1525 = stablehlo.broadcast_in_dim %1524, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1526 = stablehlo.multiply %1516, %1525 : tensor<1x4096xf32>
    %1527 = stablehlo.reshape %arg108 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1528 = stablehlo.multiply %1526, %1527 : tensor<1x4096xf32>
    %1529 = stablehlo.transpose %arg276, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1530 = stablehlo.dot %1528, %1529, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1531 = stablehlo.logistic %1530 : tensor<1x11008xf32>
    %1532 = stablehlo.multiply %1530, %1531 : tensor<1x11008xf32>
    %1533 = stablehlo.transpose %arg107, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1534 = stablehlo.dot %1528, %1533, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1535 = stablehlo.multiply %1532, %1534 : tensor<1x11008xf32>
    %1536 = stablehlo.transpose %arg106, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1537 = stablehlo.dot %1535, %1536, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %1538 = stablehlo.add %1516, %1537 : tensor<1x4096xf32>
    %1539 = stablehlo.power %1538, %1 : tensor<1x4096xf32>
    %1540 = stablehlo.reduce(%1539 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1541 = stablehlo.multiply %1540, %0 : tensor<1xf32>
    %1542 = stablehlo.reshape %1541 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1543 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1544 = stablehlo.add %1542, %1543 : tensor<1x1xf32>
    %1545 = stablehlo.rsqrt %1544 : tensor<1x1xf32>
    %1546 = stablehlo.reshape %1545 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1547 = stablehlo.broadcast_in_dim %1546, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1548 = stablehlo.multiply %1538, %1547 : tensor<1x4096xf32>
    %1549 = stablehlo.reshape %arg105 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1550 = stablehlo.multiply %1548, %1549 : tensor<1x4096xf32>
    %1551 = stablehlo.transpose %arg280, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1552 = stablehlo.dot %1550, %1551, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1553 = stablehlo.reshape %1552 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1554 = stablehlo.slice %1553 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1555 = stablehlo.reshape %1554 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1556 = stablehlo.slice %1553 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1557 = stablehlo.reshape %1556 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1558 = stablehlo.complex %1555, %1557 : tensor<1x32x64xcomplex<f32>>
    %1559 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1560 = stablehlo.multiply %1558, %1559 : tensor<1x32x64xcomplex<f32>>
    %1561 = stablehlo.real %1560 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1562 = stablehlo.reshape %1561 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1563 = stablehlo.imag %1560 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1564 = stablehlo.reshape %1563 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1565 = stablehlo.concatenate %1562, %1564, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1566 = stablehlo.reshape %1565 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %1567 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1568 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1569 = stablehlo.add %arg198, %1568 : tensor<1xi64>
    %1570 = stablehlo.select %1567, %1569, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1571 = stablehlo.reshape %1570 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1572 = stablehlo.transpose %arg278, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1573 = stablehlo.dot %1550, %1572, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1574 = stablehlo.reshape %1573 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1575 = stablehlo.slice %1574 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1576 = stablehlo.reshape %1575 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1577 = stablehlo.slice %1574 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1578 = stablehlo.reshape %1577 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1579 = stablehlo.complex %1576, %1578 : tensor<1x32x64xcomplex<f32>>
    %1580 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1581 = stablehlo.multiply %1579, %1580 : tensor<1x32x64xcomplex<f32>>
    %1582 = stablehlo.real %1581 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1583 = stablehlo.reshape %1582 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1584 = stablehlo.imag %1581 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1585 = stablehlo.reshape %1584 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1586 = stablehlo.concatenate %1583, %1585, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1587 = stablehlo.reshape %1586 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %1588 = "stablehlo.scatter"(%arg279, %1571, %1587) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1589 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1590 = "stablehlo.gather"(%1588, %1589) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1591 = stablehlo.transpose %1590, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %1592 = stablehlo.dot_general %1566, %1591, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %1593 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %1594 = stablehlo.divide %1592, %1593 : tensor<32x1x2049xf32>
    %1595 = stablehlo.reduce(%1594 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1596 = stablehlo.broadcast_in_dim %1595, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1597 = stablehlo.subtract %1594, %1596 : tensor<32x1x2049xf32>
    %1598 = stablehlo.exponential %1597 : tensor<32x1x2049xf32>
    %1599 = stablehlo.reduce(%1598 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1600 = stablehlo.broadcast_in_dim %1599, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1601 = stablehlo.divide %1598, %1600 : tensor<32x1x2049xf32>
    %1602 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1603 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1604 = stablehlo.add %arg198, %1603 : tensor<1xi64>
    %1605 = stablehlo.select %1602, %1604, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1606 = stablehlo.reshape %1605 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1607 = stablehlo.transpose %arg104, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1608 = stablehlo.dot %1550, %1607, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1609 = stablehlo.reshape %1608 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %1610 = "stablehlo.scatter"(%arg277, %1606, %1609) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1611 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1612 = "stablehlo.gather"(%1610, %1611) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1613 = stablehlo.transpose %1612, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %1614 = stablehlo.dot_general %1601, %1613, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %1615 = stablehlo.reshape %1614 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %1616 = stablehlo.transpose %arg103, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1617 = stablehlo.dot %1615, %1616, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1618 = stablehlo.add %1538, %1617 : tensor<1x4096xf32>
    %1619 = stablehlo.power %1618, %1 : tensor<1x4096xf32>
    %1620 = stablehlo.reduce(%1619 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1621 = stablehlo.multiply %1620, %0 : tensor<1xf32>
    %1622 = stablehlo.reshape %1621 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1623 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1624 = stablehlo.add %1622, %1623 : tensor<1x1xf32>
    %1625 = stablehlo.rsqrt %1624 : tensor<1x1xf32>
    %1626 = stablehlo.reshape %1625 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1627 = stablehlo.broadcast_in_dim %1626, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1628 = stablehlo.multiply %1618, %1627 : tensor<1x4096xf32>
    %1629 = stablehlo.reshape %arg102 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1630 = stablehlo.multiply %1628, %1629 : tensor<1x4096xf32>
    %1631 = stablehlo.transpose %arg281, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1632 = stablehlo.dot %1630, %1631, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1633 = stablehlo.logistic %1632 : tensor<1x11008xf32>
    %1634 = stablehlo.multiply %1632, %1633 : tensor<1x11008xf32>
    %1635 = stablehlo.transpose %arg101, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1636 = stablehlo.dot %1630, %1635, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1637 = stablehlo.multiply %1634, %1636 : tensor<1x11008xf32>
    %1638 = stablehlo.transpose %arg100, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1639 = stablehlo.dot %1637, %1638, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %1640 = stablehlo.add %1618, %1639 : tensor<1x4096xf32>
    %1641 = stablehlo.power %1640, %1 : tensor<1x4096xf32>
    %1642 = stablehlo.reduce(%1641 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1643 = stablehlo.multiply %1642, %0 : tensor<1xf32>
    %1644 = stablehlo.reshape %1643 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1645 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1646 = stablehlo.add %1644, %1645 : tensor<1x1xf32>
    %1647 = stablehlo.rsqrt %1646 : tensor<1x1xf32>
    %1648 = stablehlo.reshape %1647 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1649 = stablehlo.broadcast_in_dim %1648, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1650 = stablehlo.multiply %1640, %1649 : tensor<1x4096xf32>
    %1651 = stablehlo.reshape %arg99 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1652 = stablehlo.multiply %1650, %1651 : tensor<1x4096xf32>
    %1653 = stablehlo.transpose %arg285, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1654 = stablehlo.dot %1652, %1653, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1655 = stablehlo.reshape %1654 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1656 = stablehlo.slice %1655 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1657 = stablehlo.reshape %1656 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1658 = stablehlo.slice %1655 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1659 = stablehlo.reshape %1658 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1660 = stablehlo.complex %1657, %1659 : tensor<1x32x64xcomplex<f32>>
    %1661 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1662 = stablehlo.multiply %1660, %1661 : tensor<1x32x64xcomplex<f32>>
    %1663 = stablehlo.real %1662 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1664 = stablehlo.reshape %1663 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1665 = stablehlo.imag %1662 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1666 = stablehlo.reshape %1665 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1667 = stablehlo.concatenate %1664, %1666, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1668 = stablehlo.reshape %1667 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %1669 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1670 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1671 = stablehlo.add %arg198, %1670 : tensor<1xi64>
    %1672 = stablehlo.select %1669, %1671, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1673 = stablehlo.reshape %1672 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1674 = stablehlo.transpose %arg283, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1675 = stablehlo.dot %1652, %1674, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1676 = stablehlo.reshape %1675 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1677 = stablehlo.slice %1676 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1678 = stablehlo.reshape %1677 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1679 = stablehlo.slice %1676 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1680 = stablehlo.reshape %1679 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1681 = stablehlo.complex %1678, %1680 : tensor<1x32x64xcomplex<f32>>
    %1682 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1683 = stablehlo.multiply %1681, %1682 : tensor<1x32x64xcomplex<f32>>
    %1684 = stablehlo.real %1683 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1685 = stablehlo.reshape %1684 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1686 = stablehlo.imag %1683 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1687 = stablehlo.reshape %1686 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1688 = stablehlo.concatenate %1685, %1687, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1689 = stablehlo.reshape %1688 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %1690 = "stablehlo.scatter"(%arg284, %1673, %1689) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1691 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1692 = "stablehlo.gather"(%1690, %1691) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1693 = stablehlo.transpose %1692, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %1694 = stablehlo.dot_general %1668, %1693, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %1695 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %1696 = stablehlo.divide %1694, %1695 : tensor<32x1x2049xf32>
    %1697 = stablehlo.reduce(%1696 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1698 = stablehlo.broadcast_in_dim %1697, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1699 = stablehlo.subtract %1696, %1698 : tensor<32x1x2049xf32>
    %1700 = stablehlo.exponential %1699 : tensor<32x1x2049xf32>
    %1701 = stablehlo.reduce(%1700 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1702 = stablehlo.broadcast_in_dim %1701, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1703 = stablehlo.divide %1700, %1702 : tensor<32x1x2049xf32>
    %1704 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1705 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1706 = stablehlo.add %arg198, %1705 : tensor<1xi64>
    %1707 = stablehlo.select %1704, %1706, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1708 = stablehlo.reshape %1707 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1709 = stablehlo.transpose %arg98, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1710 = stablehlo.dot %1652, %1709, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1711 = stablehlo.reshape %1710 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %1712 = "stablehlo.scatter"(%arg282, %1708, %1711) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1713 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1714 = "stablehlo.gather"(%1712, %1713) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1715 = stablehlo.transpose %1714, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %1716 = stablehlo.dot_general %1703, %1715, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %1717 = stablehlo.reshape %1716 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %1718 = stablehlo.transpose %arg97, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1719 = stablehlo.dot %1717, %1718, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1720 = stablehlo.add %1640, %1719 : tensor<1x4096xf32>
    %1721 = stablehlo.power %1720, %1 : tensor<1x4096xf32>
    %1722 = stablehlo.reduce(%1721 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1723 = stablehlo.multiply %1722, %0 : tensor<1xf32>
    %1724 = stablehlo.reshape %1723 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1725 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1726 = stablehlo.add %1724, %1725 : tensor<1x1xf32>
    %1727 = stablehlo.rsqrt %1726 : tensor<1x1xf32>
    %1728 = stablehlo.reshape %1727 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1729 = stablehlo.broadcast_in_dim %1728, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1730 = stablehlo.multiply %1720, %1729 : tensor<1x4096xf32>
    %1731 = stablehlo.reshape %arg96 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1732 = stablehlo.multiply %1730, %1731 : tensor<1x4096xf32>
    %1733 = stablehlo.transpose %arg286, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1734 = stablehlo.dot %1732, %1733, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1735 = stablehlo.logistic %1734 : tensor<1x11008xf32>
    %1736 = stablehlo.multiply %1734, %1735 : tensor<1x11008xf32>
    %1737 = stablehlo.transpose %arg95, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1738 = stablehlo.dot %1732, %1737, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1739 = stablehlo.multiply %1736, %1738 : tensor<1x11008xf32>
    %1740 = stablehlo.transpose %arg94, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1741 = stablehlo.dot %1739, %1740, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %1742 = stablehlo.add %1720, %1741 : tensor<1x4096xf32>
    %1743 = stablehlo.power %1742, %1 : tensor<1x4096xf32>
    %1744 = stablehlo.reduce(%1743 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1745 = stablehlo.multiply %1744, %0 : tensor<1xf32>
    %1746 = stablehlo.reshape %1745 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1747 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1748 = stablehlo.add %1746, %1747 : tensor<1x1xf32>
    %1749 = stablehlo.rsqrt %1748 : tensor<1x1xf32>
    %1750 = stablehlo.reshape %1749 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1751 = stablehlo.broadcast_in_dim %1750, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1752 = stablehlo.multiply %1742, %1751 : tensor<1x4096xf32>
    %1753 = stablehlo.reshape %arg93 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1754 = stablehlo.multiply %1752, %1753 : tensor<1x4096xf32>
    %1755 = stablehlo.transpose %arg290, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1756 = stablehlo.dot %1754, %1755, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1757 = stablehlo.reshape %1756 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1758 = stablehlo.slice %1757 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1759 = stablehlo.reshape %1758 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1760 = stablehlo.slice %1757 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1761 = stablehlo.reshape %1760 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1762 = stablehlo.complex %1759, %1761 : tensor<1x32x64xcomplex<f32>>
    %1763 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1764 = stablehlo.multiply %1762, %1763 : tensor<1x32x64xcomplex<f32>>
    %1765 = stablehlo.real %1764 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1766 = stablehlo.reshape %1765 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1767 = stablehlo.imag %1764 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1768 = stablehlo.reshape %1767 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1769 = stablehlo.concatenate %1766, %1768, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1770 = stablehlo.reshape %1769 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %1771 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1772 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1773 = stablehlo.add %arg198, %1772 : tensor<1xi64>
    %1774 = stablehlo.select %1771, %1773, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1775 = stablehlo.reshape %1774 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1776 = stablehlo.transpose %arg288, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1777 = stablehlo.dot %1754, %1776, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1778 = stablehlo.reshape %1777 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1779 = stablehlo.slice %1778 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1780 = stablehlo.reshape %1779 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1781 = stablehlo.slice %1778 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1782 = stablehlo.reshape %1781 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1783 = stablehlo.complex %1780, %1782 : tensor<1x32x64xcomplex<f32>>
    %1784 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1785 = stablehlo.multiply %1783, %1784 : tensor<1x32x64xcomplex<f32>>
    %1786 = stablehlo.real %1785 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1787 = stablehlo.reshape %1786 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1788 = stablehlo.imag %1785 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1789 = stablehlo.reshape %1788 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1790 = stablehlo.concatenate %1787, %1789, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1791 = stablehlo.reshape %1790 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %1792 = "stablehlo.scatter"(%arg289, %1775, %1791) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1793 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1794 = "stablehlo.gather"(%1792, %1793) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1795 = stablehlo.transpose %1794, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %1796 = stablehlo.dot_general %1770, %1795, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %1797 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %1798 = stablehlo.divide %1796, %1797 : tensor<32x1x2049xf32>
    %1799 = stablehlo.reduce(%1798 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1800 = stablehlo.broadcast_in_dim %1799, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1801 = stablehlo.subtract %1798, %1800 : tensor<32x1x2049xf32>
    %1802 = stablehlo.exponential %1801 : tensor<32x1x2049xf32>
    %1803 = stablehlo.reduce(%1802 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1804 = stablehlo.broadcast_in_dim %1803, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1805 = stablehlo.divide %1802, %1804 : tensor<32x1x2049xf32>
    %1806 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1807 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1808 = stablehlo.add %arg198, %1807 : tensor<1xi64>
    %1809 = stablehlo.select %1806, %1808, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1810 = stablehlo.reshape %1809 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1811 = stablehlo.transpose %arg92, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1812 = stablehlo.dot %1754, %1811, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1813 = stablehlo.reshape %1812 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %1814 = "stablehlo.scatter"(%arg287, %1810, %1813) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1815 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1816 = "stablehlo.gather"(%1814, %1815) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1817 = stablehlo.transpose %1816, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %1818 = stablehlo.dot_general %1805, %1817, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %1819 = stablehlo.reshape %1818 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %1820 = stablehlo.transpose %arg91, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1821 = stablehlo.dot %1819, %1820, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1822 = stablehlo.add %1742, %1821 : tensor<1x4096xf32>
    %1823 = stablehlo.power %1822, %1 : tensor<1x4096xf32>
    %1824 = stablehlo.reduce(%1823 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1825 = stablehlo.multiply %1824, %0 : tensor<1xf32>
    %1826 = stablehlo.reshape %1825 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1827 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1828 = stablehlo.add %1826, %1827 : tensor<1x1xf32>
    %1829 = stablehlo.rsqrt %1828 : tensor<1x1xf32>
    %1830 = stablehlo.reshape %1829 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1831 = stablehlo.broadcast_in_dim %1830, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1832 = stablehlo.multiply %1822, %1831 : tensor<1x4096xf32>
    %1833 = stablehlo.reshape %arg90 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1834 = stablehlo.multiply %1832, %1833 : tensor<1x4096xf32>
    %1835 = stablehlo.transpose %arg291, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1836 = stablehlo.dot %1834, %1835, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1837 = stablehlo.logistic %1836 : tensor<1x11008xf32>
    %1838 = stablehlo.multiply %1836, %1837 : tensor<1x11008xf32>
    %1839 = stablehlo.transpose %arg89, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1840 = stablehlo.dot %1834, %1839, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1841 = stablehlo.multiply %1838, %1840 : tensor<1x11008xf32>
    %1842 = stablehlo.transpose %arg88, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1843 = stablehlo.dot %1841, %1842, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %1844 = stablehlo.add %1822, %1843 : tensor<1x4096xf32>
    %1845 = stablehlo.power %1844, %1 : tensor<1x4096xf32>
    %1846 = stablehlo.reduce(%1845 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1847 = stablehlo.multiply %1846, %0 : tensor<1xf32>
    %1848 = stablehlo.reshape %1847 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1849 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1850 = stablehlo.add %1848, %1849 : tensor<1x1xf32>
    %1851 = stablehlo.rsqrt %1850 : tensor<1x1xf32>
    %1852 = stablehlo.reshape %1851 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1853 = stablehlo.broadcast_in_dim %1852, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1854 = stablehlo.multiply %1844, %1853 : tensor<1x4096xf32>
    %1855 = stablehlo.reshape %arg87 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1856 = stablehlo.multiply %1854, %1855 : tensor<1x4096xf32>
    %1857 = stablehlo.transpose %arg295, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1858 = stablehlo.dot %1856, %1857, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1859 = stablehlo.reshape %1858 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1860 = stablehlo.slice %1859 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1861 = stablehlo.reshape %1860 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1862 = stablehlo.slice %1859 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1863 = stablehlo.reshape %1862 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1864 = stablehlo.complex %1861, %1863 : tensor<1x32x64xcomplex<f32>>
    %1865 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1866 = stablehlo.multiply %1864, %1865 : tensor<1x32x64xcomplex<f32>>
    %1867 = stablehlo.real %1866 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1868 = stablehlo.reshape %1867 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1869 = stablehlo.imag %1866 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1870 = stablehlo.reshape %1869 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1871 = stablehlo.concatenate %1868, %1870, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1872 = stablehlo.reshape %1871 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %1873 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1874 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1875 = stablehlo.add %arg198, %1874 : tensor<1xi64>
    %1876 = stablehlo.select %1873, %1875, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1877 = stablehlo.reshape %1876 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1878 = stablehlo.transpose %arg293, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1879 = stablehlo.dot %1856, %1878, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1880 = stablehlo.reshape %1879 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1881 = stablehlo.slice %1880 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1882 = stablehlo.reshape %1881 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1883 = stablehlo.slice %1880 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1884 = stablehlo.reshape %1883 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1885 = stablehlo.complex %1882, %1884 : tensor<1x32x64xcomplex<f32>>
    %1886 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1887 = stablehlo.multiply %1885, %1886 : tensor<1x32x64xcomplex<f32>>
    %1888 = stablehlo.real %1887 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1889 = stablehlo.reshape %1888 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1890 = stablehlo.imag %1887 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1891 = stablehlo.reshape %1890 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1892 = stablehlo.concatenate %1889, %1891, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1893 = stablehlo.reshape %1892 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %1894 = "stablehlo.scatter"(%arg294, %1877, %1893) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1895 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1896 = "stablehlo.gather"(%1894, %1895) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1897 = stablehlo.transpose %1896, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %1898 = stablehlo.dot_general %1872, %1897, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %1899 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %1900 = stablehlo.divide %1898, %1899 : tensor<32x1x2049xf32>
    %1901 = stablehlo.reduce(%1900 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1902 = stablehlo.broadcast_in_dim %1901, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1903 = stablehlo.subtract %1900, %1902 : tensor<32x1x2049xf32>
    %1904 = stablehlo.exponential %1903 : tensor<32x1x2049xf32>
    %1905 = stablehlo.reduce(%1904 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1906 = stablehlo.broadcast_in_dim %1905, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %1907 = stablehlo.divide %1904, %1906 : tensor<32x1x2049xf32>
    %1908 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1909 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1910 = stablehlo.add %arg198, %1909 : tensor<1xi64>
    %1911 = stablehlo.select %1908, %1910, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1912 = stablehlo.reshape %1911 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1913 = stablehlo.transpose %arg86, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1914 = stablehlo.dot %1856, %1913, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1915 = stablehlo.reshape %1914 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %1916 = "stablehlo.scatter"(%arg292, %1912, %1915) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1917 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1918 = "stablehlo.gather"(%1916, %1917) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1919 = stablehlo.transpose %1918, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %1920 = stablehlo.dot_general %1907, %1919, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %1921 = stablehlo.reshape %1920 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %1922 = stablehlo.transpose %arg85, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1923 = stablehlo.dot %1921, %1922, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1924 = stablehlo.add %1844, %1923 : tensor<1x4096xf32>
    %1925 = stablehlo.power %1924, %1 : tensor<1x4096xf32>
    %1926 = stablehlo.reduce(%1925 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1927 = stablehlo.multiply %1926, %0 : tensor<1xf32>
    %1928 = stablehlo.reshape %1927 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1929 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1930 = stablehlo.add %1928, %1929 : tensor<1x1xf32>
    %1931 = stablehlo.rsqrt %1930 : tensor<1x1xf32>
    %1932 = stablehlo.reshape %1931 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1933 = stablehlo.broadcast_in_dim %1932, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1934 = stablehlo.multiply %1924, %1933 : tensor<1x4096xf32>
    %1935 = stablehlo.reshape %arg84 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1936 = stablehlo.multiply %1934, %1935 : tensor<1x4096xf32>
    %1937 = stablehlo.transpose %arg296, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1938 = stablehlo.dot %1936, %1937, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1939 = stablehlo.logistic %1938 : tensor<1x11008xf32>
    %1940 = stablehlo.multiply %1938, %1939 : tensor<1x11008xf32>
    %1941 = stablehlo.transpose %arg83, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1942 = stablehlo.dot %1936, %1941, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %1943 = stablehlo.multiply %1940, %1942 : tensor<1x11008xf32>
    %1944 = stablehlo.transpose %arg82, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1945 = stablehlo.dot %1943, %1944, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %1946 = stablehlo.add %1924, %1945 : tensor<1x4096xf32>
    %1947 = stablehlo.power %1946, %1 : tensor<1x4096xf32>
    %1948 = stablehlo.reduce(%1947 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %1949 = stablehlo.multiply %1948, %0 : tensor<1xf32>
    %1950 = stablehlo.reshape %1949 : (tensor<1xf32>) -> tensor<1x1xf32>
    %1951 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %1952 = stablehlo.add %1950, %1951 : tensor<1x1xf32>
    %1953 = stablehlo.rsqrt %1952 : tensor<1x1xf32>
    %1954 = stablehlo.reshape %1953 : (tensor<1x1xf32>) -> tensor<1xf32>
    %1955 = stablehlo.broadcast_in_dim %1954, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %1956 = stablehlo.multiply %1946, %1955 : tensor<1x4096xf32>
    %1957 = stablehlo.reshape %arg81 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %1958 = stablehlo.multiply %1956, %1957 : tensor<1x4096xf32>
    %1959 = stablehlo.transpose %arg300, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1960 = stablehlo.dot %1958, %1959, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1961 = stablehlo.reshape %1960 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1962 = stablehlo.slice %1961 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1963 = stablehlo.reshape %1962 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1964 = stablehlo.slice %1961 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1965 = stablehlo.reshape %1964 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1966 = stablehlo.complex %1963, %1965 : tensor<1x32x64xcomplex<f32>>
    %1967 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1968 = stablehlo.multiply %1966, %1967 : tensor<1x32x64xcomplex<f32>>
    %1969 = stablehlo.real %1968 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1970 = stablehlo.reshape %1969 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1971 = stablehlo.imag %1968 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1972 = stablehlo.reshape %1971 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1973 = stablehlo.concatenate %1970, %1972, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1974 = stablehlo.reshape %1973 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %1975 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1976 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %1977 = stablehlo.add %arg198, %1976 : tensor<1xi64>
    %1978 = stablehlo.select %1975, %1977, %arg198 : tensor<1xi1>, tensor<1xi64>
    %1979 = stablehlo.reshape %1978 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1980 = stablehlo.transpose %arg298, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1981 = stablehlo.dot %1958, %1980, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %1982 = stablehlo.reshape %1981 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %1983 = stablehlo.slice %1982 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1984 = stablehlo.reshape %1983 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1985 = stablehlo.slice %1982 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %1986 = stablehlo.reshape %1985 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %1987 = stablehlo.complex %1984, %1986 : tensor<1x32x64xcomplex<f32>>
    %1988 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %1989 = stablehlo.multiply %1987, %1988 : tensor<1x32x64xcomplex<f32>>
    %1990 = stablehlo.real %1989 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1991 = stablehlo.reshape %1990 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1992 = stablehlo.imag %1989 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %1993 = stablehlo.reshape %1992 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %1994 = stablehlo.concatenate %1991, %1993, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %1995 = stablehlo.reshape %1994 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %1996 = "stablehlo.scatter"(%arg299, %1979, %1995) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %1997 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %1998 = "stablehlo.gather"(%1996, %1997) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %1999 = stablehlo.transpose %1998, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %2000 = stablehlo.dot_general %1974, %1999, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %2001 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %2002 = stablehlo.divide %2000, %2001 : tensor<32x1x2049xf32>
    %2003 = stablehlo.reduce(%2002 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2004 = stablehlo.broadcast_in_dim %2003, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2005 = stablehlo.subtract %2002, %2004 : tensor<32x1x2049xf32>
    %2006 = stablehlo.exponential %2005 : tensor<32x1x2049xf32>
    %2007 = stablehlo.reduce(%2006 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2008 = stablehlo.broadcast_in_dim %2007, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2009 = stablehlo.divide %2006, %2008 : tensor<32x1x2049xf32>
    %2010 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2011 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2012 = stablehlo.add %arg198, %2011 : tensor<1xi64>
    %2013 = stablehlo.select %2010, %2012, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2014 = stablehlo.reshape %2013 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2015 = stablehlo.transpose %arg80, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2016 = stablehlo.dot %1958, %2015, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2017 = stablehlo.reshape %2016 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %2018 = "stablehlo.scatter"(%arg297, %2014, %2017) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2019 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2020 = "stablehlo.gather"(%2018, %2019) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2021 = stablehlo.transpose %2020, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %2022 = stablehlo.dot_general %2009, %2021, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %2023 = stablehlo.reshape %2022 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %2024 = stablehlo.transpose %arg79, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2025 = stablehlo.dot %2023, %2024, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2026 = stablehlo.add %1946, %2025 : tensor<1x4096xf32>
    %2027 = stablehlo.power %2026, %1 : tensor<1x4096xf32>
    %2028 = stablehlo.reduce(%2027 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2029 = stablehlo.multiply %2028, %0 : tensor<1xf32>
    %2030 = stablehlo.reshape %2029 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2031 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2032 = stablehlo.add %2030, %2031 : tensor<1x1xf32>
    %2033 = stablehlo.rsqrt %2032 : tensor<1x1xf32>
    %2034 = stablehlo.reshape %2033 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2035 = stablehlo.broadcast_in_dim %2034, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2036 = stablehlo.multiply %2026, %2035 : tensor<1x4096xf32>
    %2037 = stablehlo.reshape %arg78 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2038 = stablehlo.multiply %2036, %2037 : tensor<1x4096xf32>
    %2039 = stablehlo.transpose %arg301, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2040 = stablehlo.dot %2038, %2039, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2041 = stablehlo.logistic %2040 : tensor<1x11008xf32>
    %2042 = stablehlo.multiply %2040, %2041 : tensor<1x11008xf32>
    %2043 = stablehlo.transpose %arg77, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2044 = stablehlo.dot %2038, %2043, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2045 = stablehlo.multiply %2042, %2044 : tensor<1x11008xf32>
    %2046 = stablehlo.transpose %arg76, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2047 = stablehlo.dot %2045, %2046, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %2048 = stablehlo.add %2026, %2047 : tensor<1x4096xf32>
    %2049 = stablehlo.power %2048, %1 : tensor<1x4096xf32>
    %2050 = stablehlo.reduce(%2049 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2051 = stablehlo.multiply %2050, %0 : tensor<1xf32>
    %2052 = stablehlo.reshape %2051 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2053 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2054 = stablehlo.add %2052, %2053 : tensor<1x1xf32>
    %2055 = stablehlo.rsqrt %2054 : tensor<1x1xf32>
    %2056 = stablehlo.reshape %2055 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2057 = stablehlo.broadcast_in_dim %2056, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2058 = stablehlo.multiply %2048, %2057 : tensor<1x4096xf32>
    %2059 = stablehlo.reshape %arg75 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2060 = stablehlo.multiply %2058, %2059 : tensor<1x4096xf32>
    %2061 = stablehlo.transpose %arg305, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2062 = stablehlo.dot %2060, %2061, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2063 = stablehlo.reshape %2062 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2064 = stablehlo.slice %2063 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2065 = stablehlo.reshape %2064 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2066 = stablehlo.slice %2063 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2067 = stablehlo.reshape %2066 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2068 = stablehlo.complex %2065, %2067 : tensor<1x32x64xcomplex<f32>>
    %2069 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2070 = stablehlo.multiply %2068, %2069 : tensor<1x32x64xcomplex<f32>>
    %2071 = stablehlo.real %2070 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2072 = stablehlo.reshape %2071 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2073 = stablehlo.imag %2070 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2074 = stablehlo.reshape %2073 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2075 = stablehlo.concatenate %2072, %2074, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2076 = stablehlo.reshape %2075 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %2077 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2078 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2079 = stablehlo.add %arg198, %2078 : tensor<1xi64>
    %2080 = stablehlo.select %2077, %2079, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2081 = stablehlo.reshape %2080 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2082 = stablehlo.transpose %arg303, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2083 = stablehlo.dot %2060, %2082, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2084 = stablehlo.reshape %2083 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2085 = stablehlo.slice %2084 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2086 = stablehlo.reshape %2085 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2087 = stablehlo.slice %2084 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2088 = stablehlo.reshape %2087 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2089 = stablehlo.complex %2086, %2088 : tensor<1x32x64xcomplex<f32>>
    %2090 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2091 = stablehlo.multiply %2089, %2090 : tensor<1x32x64xcomplex<f32>>
    %2092 = stablehlo.real %2091 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2093 = stablehlo.reshape %2092 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2094 = stablehlo.imag %2091 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2095 = stablehlo.reshape %2094 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2096 = stablehlo.concatenate %2093, %2095, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2097 = stablehlo.reshape %2096 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %2098 = "stablehlo.scatter"(%arg304, %2081, %2097) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2099 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2100 = "stablehlo.gather"(%2098, %2099) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2101 = stablehlo.transpose %2100, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %2102 = stablehlo.dot_general %2076, %2101, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %2103 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %2104 = stablehlo.divide %2102, %2103 : tensor<32x1x2049xf32>
    %2105 = stablehlo.reduce(%2104 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2106 = stablehlo.broadcast_in_dim %2105, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2107 = stablehlo.subtract %2104, %2106 : tensor<32x1x2049xf32>
    %2108 = stablehlo.exponential %2107 : tensor<32x1x2049xf32>
    %2109 = stablehlo.reduce(%2108 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2110 = stablehlo.broadcast_in_dim %2109, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2111 = stablehlo.divide %2108, %2110 : tensor<32x1x2049xf32>
    %2112 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2113 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2114 = stablehlo.add %arg198, %2113 : tensor<1xi64>
    %2115 = stablehlo.select %2112, %2114, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2116 = stablehlo.reshape %2115 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2117 = stablehlo.transpose %arg74, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2118 = stablehlo.dot %2060, %2117, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2119 = stablehlo.reshape %2118 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %2120 = "stablehlo.scatter"(%arg302, %2116, %2119) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2121 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2122 = "stablehlo.gather"(%2120, %2121) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2123 = stablehlo.transpose %2122, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %2124 = stablehlo.dot_general %2111, %2123, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %2125 = stablehlo.reshape %2124 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %2126 = stablehlo.transpose %arg73, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2127 = stablehlo.dot %2125, %2126, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2128 = stablehlo.add %2048, %2127 : tensor<1x4096xf32>
    %2129 = stablehlo.power %2128, %1 : tensor<1x4096xf32>
    %2130 = stablehlo.reduce(%2129 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2131 = stablehlo.multiply %2130, %0 : tensor<1xf32>
    %2132 = stablehlo.reshape %2131 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2133 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2134 = stablehlo.add %2132, %2133 : tensor<1x1xf32>
    %2135 = stablehlo.rsqrt %2134 : tensor<1x1xf32>
    %2136 = stablehlo.reshape %2135 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2137 = stablehlo.broadcast_in_dim %2136, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2138 = stablehlo.multiply %2128, %2137 : tensor<1x4096xf32>
    %2139 = stablehlo.reshape %arg72 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2140 = stablehlo.multiply %2138, %2139 : tensor<1x4096xf32>
    %2141 = stablehlo.transpose %arg306, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2142 = stablehlo.dot %2140, %2141, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2143 = stablehlo.logistic %2142 : tensor<1x11008xf32>
    %2144 = stablehlo.multiply %2142, %2143 : tensor<1x11008xf32>
    %2145 = stablehlo.transpose %arg71, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2146 = stablehlo.dot %2140, %2145, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2147 = stablehlo.multiply %2144, %2146 : tensor<1x11008xf32>
    %2148 = stablehlo.transpose %arg70, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2149 = stablehlo.dot %2147, %2148, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %2150 = stablehlo.add %2128, %2149 : tensor<1x4096xf32>
    %2151 = stablehlo.power %2150, %1 : tensor<1x4096xf32>
    %2152 = stablehlo.reduce(%2151 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2153 = stablehlo.multiply %2152, %0 : tensor<1xf32>
    %2154 = stablehlo.reshape %2153 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2155 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2156 = stablehlo.add %2154, %2155 : tensor<1x1xf32>
    %2157 = stablehlo.rsqrt %2156 : tensor<1x1xf32>
    %2158 = stablehlo.reshape %2157 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2159 = stablehlo.broadcast_in_dim %2158, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2160 = stablehlo.multiply %2150, %2159 : tensor<1x4096xf32>
    %2161 = stablehlo.reshape %arg69 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2162 = stablehlo.multiply %2160, %2161 : tensor<1x4096xf32>
    %2163 = stablehlo.transpose %arg310, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2164 = stablehlo.dot %2162, %2163, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2165 = stablehlo.reshape %2164 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2166 = stablehlo.slice %2165 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2167 = stablehlo.reshape %2166 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2168 = stablehlo.slice %2165 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2169 = stablehlo.reshape %2168 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2170 = stablehlo.complex %2167, %2169 : tensor<1x32x64xcomplex<f32>>
    %2171 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2172 = stablehlo.multiply %2170, %2171 : tensor<1x32x64xcomplex<f32>>
    %2173 = stablehlo.real %2172 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2174 = stablehlo.reshape %2173 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2175 = stablehlo.imag %2172 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2176 = stablehlo.reshape %2175 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2177 = stablehlo.concatenate %2174, %2176, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2178 = stablehlo.reshape %2177 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %2179 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2180 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2181 = stablehlo.add %arg198, %2180 : tensor<1xi64>
    %2182 = stablehlo.select %2179, %2181, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2183 = stablehlo.reshape %2182 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2184 = stablehlo.transpose %arg308, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2185 = stablehlo.dot %2162, %2184, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2186 = stablehlo.reshape %2185 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2187 = stablehlo.slice %2186 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2188 = stablehlo.reshape %2187 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2189 = stablehlo.slice %2186 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2190 = stablehlo.reshape %2189 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2191 = stablehlo.complex %2188, %2190 : tensor<1x32x64xcomplex<f32>>
    %2192 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2193 = stablehlo.multiply %2191, %2192 : tensor<1x32x64xcomplex<f32>>
    %2194 = stablehlo.real %2193 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2195 = stablehlo.reshape %2194 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2196 = stablehlo.imag %2193 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2197 = stablehlo.reshape %2196 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2198 = stablehlo.concatenate %2195, %2197, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2199 = stablehlo.reshape %2198 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %2200 = "stablehlo.scatter"(%arg309, %2183, %2199) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2201 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2202 = "stablehlo.gather"(%2200, %2201) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2203 = stablehlo.transpose %2202, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %2204 = stablehlo.dot_general %2178, %2203, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %2205 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %2206 = stablehlo.divide %2204, %2205 : tensor<32x1x2049xf32>
    %2207 = stablehlo.reduce(%2206 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2208 = stablehlo.broadcast_in_dim %2207, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2209 = stablehlo.subtract %2206, %2208 : tensor<32x1x2049xf32>
    %2210 = stablehlo.exponential %2209 : tensor<32x1x2049xf32>
    %2211 = stablehlo.reduce(%2210 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2212 = stablehlo.broadcast_in_dim %2211, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2213 = stablehlo.divide %2210, %2212 : tensor<32x1x2049xf32>
    %2214 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2215 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2216 = stablehlo.add %arg198, %2215 : tensor<1xi64>
    %2217 = stablehlo.select %2214, %2216, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2218 = stablehlo.reshape %2217 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2219 = stablehlo.transpose %arg68, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2220 = stablehlo.dot %2162, %2219, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2221 = stablehlo.reshape %2220 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %2222 = "stablehlo.scatter"(%arg307, %2218, %2221) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2223 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2224 = "stablehlo.gather"(%2222, %2223) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2225 = stablehlo.transpose %2224, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %2226 = stablehlo.dot_general %2213, %2225, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %2227 = stablehlo.reshape %2226 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %2228 = stablehlo.transpose %arg67, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2229 = stablehlo.dot %2227, %2228, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2230 = stablehlo.add %2150, %2229 : tensor<1x4096xf32>
    %2231 = stablehlo.power %2230, %1 : tensor<1x4096xf32>
    %2232 = stablehlo.reduce(%2231 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2233 = stablehlo.multiply %2232, %0 : tensor<1xf32>
    %2234 = stablehlo.reshape %2233 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2235 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2236 = stablehlo.add %2234, %2235 : tensor<1x1xf32>
    %2237 = stablehlo.rsqrt %2236 : tensor<1x1xf32>
    %2238 = stablehlo.reshape %2237 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2239 = stablehlo.broadcast_in_dim %2238, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2240 = stablehlo.multiply %2230, %2239 : tensor<1x4096xf32>
    %2241 = stablehlo.reshape %arg66 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2242 = stablehlo.multiply %2240, %2241 : tensor<1x4096xf32>
    %2243 = stablehlo.transpose %arg311, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2244 = stablehlo.dot %2242, %2243, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2245 = stablehlo.logistic %2244 : tensor<1x11008xf32>
    %2246 = stablehlo.multiply %2244, %2245 : tensor<1x11008xf32>
    %2247 = stablehlo.transpose %arg65, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2248 = stablehlo.dot %2242, %2247, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2249 = stablehlo.multiply %2246, %2248 : tensor<1x11008xf32>
    %2250 = stablehlo.transpose %arg64, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2251 = stablehlo.dot %2249, %2250, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %2252 = stablehlo.add %2230, %2251 : tensor<1x4096xf32>
    %2253 = stablehlo.power %2252, %1 : tensor<1x4096xf32>
    %2254 = stablehlo.reduce(%2253 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2255 = stablehlo.multiply %2254, %0 : tensor<1xf32>
    %2256 = stablehlo.reshape %2255 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2257 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2258 = stablehlo.add %2256, %2257 : tensor<1x1xf32>
    %2259 = stablehlo.rsqrt %2258 : tensor<1x1xf32>
    %2260 = stablehlo.reshape %2259 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2261 = stablehlo.broadcast_in_dim %2260, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2262 = stablehlo.multiply %2252, %2261 : tensor<1x4096xf32>
    %2263 = stablehlo.reshape %arg63 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2264 = stablehlo.multiply %2262, %2263 : tensor<1x4096xf32>
    %2265 = stablehlo.transpose %arg315, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2266 = stablehlo.dot %2264, %2265, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2267 = stablehlo.reshape %2266 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2268 = stablehlo.slice %2267 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2269 = stablehlo.reshape %2268 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2270 = stablehlo.slice %2267 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2271 = stablehlo.reshape %2270 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2272 = stablehlo.complex %2269, %2271 : tensor<1x32x64xcomplex<f32>>
    %2273 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2274 = stablehlo.multiply %2272, %2273 : tensor<1x32x64xcomplex<f32>>
    %2275 = stablehlo.real %2274 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2276 = stablehlo.reshape %2275 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2277 = stablehlo.imag %2274 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2278 = stablehlo.reshape %2277 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2279 = stablehlo.concatenate %2276, %2278, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2280 = stablehlo.reshape %2279 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %2281 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2282 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2283 = stablehlo.add %arg198, %2282 : tensor<1xi64>
    %2284 = stablehlo.select %2281, %2283, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2285 = stablehlo.reshape %2284 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2286 = stablehlo.transpose %arg313, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2287 = stablehlo.dot %2264, %2286, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2288 = stablehlo.reshape %2287 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2289 = stablehlo.slice %2288 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2290 = stablehlo.reshape %2289 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2291 = stablehlo.slice %2288 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2292 = stablehlo.reshape %2291 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2293 = stablehlo.complex %2290, %2292 : tensor<1x32x64xcomplex<f32>>
    %2294 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2295 = stablehlo.multiply %2293, %2294 : tensor<1x32x64xcomplex<f32>>
    %2296 = stablehlo.real %2295 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2297 = stablehlo.reshape %2296 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2298 = stablehlo.imag %2295 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2299 = stablehlo.reshape %2298 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2300 = stablehlo.concatenate %2297, %2299, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2301 = stablehlo.reshape %2300 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %2302 = "stablehlo.scatter"(%arg314, %2285, %2301) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2303 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2304 = "stablehlo.gather"(%2302, %2303) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2305 = stablehlo.transpose %2304, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %2306 = stablehlo.dot_general %2280, %2305, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %2307 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %2308 = stablehlo.divide %2306, %2307 : tensor<32x1x2049xf32>
    %2309 = stablehlo.reduce(%2308 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2310 = stablehlo.broadcast_in_dim %2309, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2311 = stablehlo.subtract %2308, %2310 : tensor<32x1x2049xf32>
    %2312 = stablehlo.exponential %2311 : tensor<32x1x2049xf32>
    %2313 = stablehlo.reduce(%2312 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2314 = stablehlo.broadcast_in_dim %2313, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2315 = stablehlo.divide %2312, %2314 : tensor<32x1x2049xf32>
    %2316 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2317 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2318 = stablehlo.add %arg198, %2317 : tensor<1xi64>
    %2319 = stablehlo.select %2316, %2318, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2320 = stablehlo.reshape %2319 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2321 = stablehlo.transpose %arg62, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2322 = stablehlo.dot %2264, %2321, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2323 = stablehlo.reshape %2322 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %2324 = "stablehlo.scatter"(%arg312, %2320, %2323) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2325 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2326 = "stablehlo.gather"(%2324, %2325) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2327 = stablehlo.transpose %2326, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %2328 = stablehlo.dot_general %2315, %2327, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %2329 = stablehlo.reshape %2328 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %2330 = stablehlo.transpose %arg61, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2331 = stablehlo.dot %2329, %2330, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2332 = stablehlo.add %2252, %2331 : tensor<1x4096xf32>
    %2333 = stablehlo.power %2332, %1 : tensor<1x4096xf32>
    %2334 = stablehlo.reduce(%2333 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2335 = stablehlo.multiply %2334, %0 : tensor<1xf32>
    %2336 = stablehlo.reshape %2335 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2337 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2338 = stablehlo.add %2336, %2337 : tensor<1x1xf32>
    %2339 = stablehlo.rsqrt %2338 : tensor<1x1xf32>
    %2340 = stablehlo.reshape %2339 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2341 = stablehlo.broadcast_in_dim %2340, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2342 = stablehlo.multiply %2332, %2341 : tensor<1x4096xf32>
    %2343 = stablehlo.reshape %arg60 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2344 = stablehlo.multiply %2342, %2343 : tensor<1x4096xf32>
    %2345 = stablehlo.transpose %arg316, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2346 = stablehlo.dot %2344, %2345, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2347 = stablehlo.logistic %2346 : tensor<1x11008xf32>
    %2348 = stablehlo.multiply %2346, %2347 : tensor<1x11008xf32>
    %2349 = stablehlo.transpose %arg59, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2350 = stablehlo.dot %2344, %2349, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2351 = stablehlo.multiply %2348, %2350 : tensor<1x11008xf32>
    %2352 = stablehlo.transpose %arg58, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2353 = stablehlo.dot %2351, %2352, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %2354 = stablehlo.add %2332, %2353 : tensor<1x4096xf32>
    %2355 = stablehlo.power %2354, %1 : tensor<1x4096xf32>
    %2356 = stablehlo.reduce(%2355 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2357 = stablehlo.multiply %2356, %0 : tensor<1xf32>
    %2358 = stablehlo.reshape %2357 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2359 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2360 = stablehlo.add %2358, %2359 : tensor<1x1xf32>
    %2361 = stablehlo.rsqrt %2360 : tensor<1x1xf32>
    %2362 = stablehlo.reshape %2361 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2363 = stablehlo.broadcast_in_dim %2362, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2364 = stablehlo.multiply %2354, %2363 : tensor<1x4096xf32>
    %2365 = stablehlo.reshape %arg57 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2366 = stablehlo.multiply %2364, %2365 : tensor<1x4096xf32>
    %2367 = stablehlo.transpose %arg320, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2368 = stablehlo.dot %2366, %2367, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2369 = stablehlo.reshape %2368 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2370 = stablehlo.slice %2369 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2371 = stablehlo.reshape %2370 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2372 = stablehlo.slice %2369 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2373 = stablehlo.reshape %2372 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2374 = stablehlo.complex %2371, %2373 : tensor<1x32x64xcomplex<f32>>
    %2375 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2376 = stablehlo.multiply %2374, %2375 : tensor<1x32x64xcomplex<f32>>
    %2377 = stablehlo.real %2376 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2378 = stablehlo.reshape %2377 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2379 = stablehlo.imag %2376 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2380 = stablehlo.reshape %2379 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2381 = stablehlo.concatenate %2378, %2380, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2382 = stablehlo.reshape %2381 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %2383 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2384 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2385 = stablehlo.add %arg198, %2384 : tensor<1xi64>
    %2386 = stablehlo.select %2383, %2385, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2387 = stablehlo.reshape %2386 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2388 = stablehlo.transpose %arg318, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2389 = stablehlo.dot %2366, %2388, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2390 = stablehlo.reshape %2389 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2391 = stablehlo.slice %2390 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2392 = stablehlo.reshape %2391 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2393 = stablehlo.slice %2390 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2394 = stablehlo.reshape %2393 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2395 = stablehlo.complex %2392, %2394 : tensor<1x32x64xcomplex<f32>>
    %2396 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2397 = stablehlo.multiply %2395, %2396 : tensor<1x32x64xcomplex<f32>>
    %2398 = stablehlo.real %2397 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2399 = stablehlo.reshape %2398 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2400 = stablehlo.imag %2397 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2401 = stablehlo.reshape %2400 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2402 = stablehlo.concatenate %2399, %2401, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2403 = stablehlo.reshape %2402 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %2404 = "stablehlo.scatter"(%arg319, %2387, %2403) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2405 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2406 = "stablehlo.gather"(%2404, %2405) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2407 = stablehlo.transpose %2406, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %2408 = stablehlo.dot_general %2382, %2407, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %2409 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %2410 = stablehlo.divide %2408, %2409 : tensor<32x1x2049xf32>
    %2411 = stablehlo.reduce(%2410 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2412 = stablehlo.broadcast_in_dim %2411, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2413 = stablehlo.subtract %2410, %2412 : tensor<32x1x2049xf32>
    %2414 = stablehlo.exponential %2413 : tensor<32x1x2049xf32>
    %2415 = stablehlo.reduce(%2414 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2416 = stablehlo.broadcast_in_dim %2415, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2417 = stablehlo.divide %2414, %2416 : tensor<32x1x2049xf32>
    %2418 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2419 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2420 = stablehlo.add %arg198, %2419 : tensor<1xi64>
    %2421 = stablehlo.select %2418, %2420, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2422 = stablehlo.reshape %2421 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2423 = stablehlo.transpose %arg56, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2424 = stablehlo.dot %2366, %2423, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2425 = stablehlo.reshape %2424 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %2426 = "stablehlo.scatter"(%arg317, %2422, %2425) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2427 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2428 = "stablehlo.gather"(%2426, %2427) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2429 = stablehlo.transpose %2428, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %2430 = stablehlo.dot_general %2417, %2429, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %2431 = stablehlo.reshape %2430 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %2432 = stablehlo.transpose %arg55, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2433 = stablehlo.dot %2431, %2432, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2434 = stablehlo.add %2354, %2433 : tensor<1x4096xf32>
    %2435 = stablehlo.power %2434, %1 : tensor<1x4096xf32>
    %2436 = stablehlo.reduce(%2435 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2437 = stablehlo.multiply %2436, %0 : tensor<1xf32>
    %2438 = stablehlo.reshape %2437 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2439 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2440 = stablehlo.add %2438, %2439 : tensor<1x1xf32>
    %2441 = stablehlo.rsqrt %2440 : tensor<1x1xf32>
    %2442 = stablehlo.reshape %2441 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2443 = stablehlo.broadcast_in_dim %2442, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2444 = stablehlo.multiply %2434, %2443 : tensor<1x4096xf32>
    %2445 = stablehlo.reshape %arg54 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2446 = stablehlo.multiply %2444, %2445 : tensor<1x4096xf32>
    %2447 = stablehlo.transpose %arg321, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2448 = stablehlo.dot %2446, %2447, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2449 = stablehlo.logistic %2448 : tensor<1x11008xf32>
    %2450 = stablehlo.multiply %2448, %2449 : tensor<1x11008xf32>
    %2451 = stablehlo.transpose %arg53, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2452 = stablehlo.dot %2446, %2451, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2453 = stablehlo.multiply %2450, %2452 : tensor<1x11008xf32>
    %2454 = stablehlo.transpose %arg52, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2455 = stablehlo.dot %2453, %2454, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %2456 = stablehlo.add %2434, %2455 : tensor<1x4096xf32>
    %2457 = stablehlo.power %2456, %1 : tensor<1x4096xf32>
    %2458 = stablehlo.reduce(%2457 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2459 = stablehlo.multiply %2458, %0 : tensor<1xf32>
    %2460 = stablehlo.reshape %2459 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2461 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2462 = stablehlo.add %2460, %2461 : tensor<1x1xf32>
    %2463 = stablehlo.rsqrt %2462 : tensor<1x1xf32>
    %2464 = stablehlo.reshape %2463 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2465 = stablehlo.broadcast_in_dim %2464, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2466 = stablehlo.multiply %2456, %2465 : tensor<1x4096xf32>
    %2467 = stablehlo.reshape %arg51 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2468 = stablehlo.multiply %2466, %2467 : tensor<1x4096xf32>
    %2469 = stablehlo.transpose %arg325, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2470 = stablehlo.dot %2468, %2469, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2471 = stablehlo.reshape %2470 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2472 = stablehlo.slice %2471 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2473 = stablehlo.reshape %2472 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2474 = stablehlo.slice %2471 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2475 = stablehlo.reshape %2474 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2476 = stablehlo.complex %2473, %2475 : tensor<1x32x64xcomplex<f32>>
    %2477 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2478 = stablehlo.multiply %2476, %2477 : tensor<1x32x64xcomplex<f32>>
    %2479 = stablehlo.real %2478 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2480 = stablehlo.reshape %2479 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2481 = stablehlo.imag %2478 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2482 = stablehlo.reshape %2481 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2483 = stablehlo.concatenate %2480, %2482, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2484 = stablehlo.reshape %2483 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %2485 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2486 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2487 = stablehlo.add %arg198, %2486 : tensor<1xi64>
    %2488 = stablehlo.select %2485, %2487, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2489 = stablehlo.reshape %2488 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2490 = stablehlo.transpose %arg323, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2491 = stablehlo.dot %2468, %2490, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2492 = stablehlo.reshape %2491 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2493 = stablehlo.slice %2492 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2494 = stablehlo.reshape %2493 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2495 = stablehlo.slice %2492 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2496 = stablehlo.reshape %2495 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2497 = stablehlo.complex %2494, %2496 : tensor<1x32x64xcomplex<f32>>
    %2498 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2499 = stablehlo.multiply %2497, %2498 : tensor<1x32x64xcomplex<f32>>
    %2500 = stablehlo.real %2499 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2501 = stablehlo.reshape %2500 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2502 = stablehlo.imag %2499 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2503 = stablehlo.reshape %2502 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2504 = stablehlo.concatenate %2501, %2503, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2505 = stablehlo.reshape %2504 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %2506 = "stablehlo.scatter"(%arg324, %2489, %2505) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2507 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2508 = "stablehlo.gather"(%2506, %2507) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2509 = stablehlo.transpose %2508, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %2510 = stablehlo.dot_general %2484, %2509, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %2511 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %2512 = stablehlo.divide %2510, %2511 : tensor<32x1x2049xf32>
    %2513 = stablehlo.reduce(%2512 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2514 = stablehlo.broadcast_in_dim %2513, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2515 = stablehlo.subtract %2512, %2514 : tensor<32x1x2049xf32>
    %2516 = stablehlo.exponential %2515 : tensor<32x1x2049xf32>
    %2517 = stablehlo.reduce(%2516 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2518 = stablehlo.broadcast_in_dim %2517, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2519 = stablehlo.divide %2516, %2518 : tensor<32x1x2049xf32>
    %2520 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2521 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2522 = stablehlo.add %arg198, %2521 : tensor<1xi64>
    %2523 = stablehlo.select %2520, %2522, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2524 = stablehlo.reshape %2523 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2525 = stablehlo.transpose %arg50, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2526 = stablehlo.dot %2468, %2525, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2527 = stablehlo.reshape %2526 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %2528 = "stablehlo.scatter"(%arg322, %2524, %2527) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2529 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2530 = "stablehlo.gather"(%2528, %2529) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2531 = stablehlo.transpose %2530, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %2532 = stablehlo.dot_general %2519, %2531, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %2533 = stablehlo.reshape %2532 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %2534 = stablehlo.transpose %arg49, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2535 = stablehlo.dot %2533, %2534, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2536 = stablehlo.add %2456, %2535 : tensor<1x4096xf32>
    %2537 = stablehlo.power %2536, %1 : tensor<1x4096xf32>
    %2538 = stablehlo.reduce(%2537 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2539 = stablehlo.multiply %2538, %0 : tensor<1xf32>
    %2540 = stablehlo.reshape %2539 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2541 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2542 = stablehlo.add %2540, %2541 : tensor<1x1xf32>
    %2543 = stablehlo.rsqrt %2542 : tensor<1x1xf32>
    %2544 = stablehlo.reshape %2543 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2545 = stablehlo.broadcast_in_dim %2544, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2546 = stablehlo.multiply %2536, %2545 : tensor<1x4096xf32>
    %2547 = stablehlo.reshape %arg48 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2548 = stablehlo.multiply %2546, %2547 : tensor<1x4096xf32>
    %2549 = stablehlo.transpose %arg326, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2550 = stablehlo.dot %2548, %2549, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2551 = stablehlo.logistic %2550 : tensor<1x11008xf32>
    %2552 = stablehlo.multiply %2550, %2551 : tensor<1x11008xf32>
    %2553 = stablehlo.transpose %arg47, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2554 = stablehlo.dot %2548, %2553, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2555 = stablehlo.multiply %2552, %2554 : tensor<1x11008xf32>
    %2556 = stablehlo.transpose %arg46, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2557 = stablehlo.dot %2555, %2556, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %2558 = stablehlo.add %2536, %2557 : tensor<1x4096xf32>
    %2559 = stablehlo.power %2558, %1 : tensor<1x4096xf32>
    %2560 = stablehlo.reduce(%2559 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2561 = stablehlo.multiply %2560, %0 : tensor<1xf32>
    %2562 = stablehlo.reshape %2561 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2563 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2564 = stablehlo.add %2562, %2563 : tensor<1x1xf32>
    %2565 = stablehlo.rsqrt %2564 : tensor<1x1xf32>
    %2566 = stablehlo.reshape %2565 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2567 = stablehlo.broadcast_in_dim %2566, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2568 = stablehlo.multiply %2558, %2567 : tensor<1x4096xf32>
    %2569 = stablehlo.reshape %arg45 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2570 = stablehlo.multiply %2568, %2569 : tensor<1x4096xf32>
    %2571 = stablehlo.transpose %arg330, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2572 = stablehlo.dot %2570, %2571, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2573 = stablehlo.reshape %2572 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2574 = stablehlo.slice %2573 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2575 = stablehlo.reshape %2574 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2576 = stablehlo.slice %2573 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2577 = stablehlo.reshape %2576 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2578 = stablehlo.complex %2575, %2577 : tensor<1x32x64xcomplex<f32>>
    %2579 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2580 = stablehlo.multiply %2578, %2579 : tensor<1x32x64xcomplex<f32>>
    %2581 = stablehlo.real %2580 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2582 = stablehlo.reshape %2581 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2583 = stablehlo.imag %2580 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2584 = stablehlo.reshape %2583 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2585 = stablehlo.concatenate %2582, %2584, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2586 = stablehlo.reshape %2585 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %2587 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2588 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2589 = stablehlo.add %arg198, %2588 : tensor<1xi64>
    %2590 = stablehlo.select %2587, %2589, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2591 = stablehlo.reshape %2590 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2592 = stablehlo.transpose %arg328, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2593 = stablehlo.dot %2570, %2592, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2594 = stablehlo.reshape %2593 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2595 = stablehlo.slice %2594 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2596 = stablehlo.reshape %2595 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2597 = stablehlo.slice %2594 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2598 = stablehlo.reshape %2597 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2599 = stablehlo.complex %2596, %2598 : tensor<1x32x64xcomplex<f32>>
    %2600 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2601 = stablehlo.multiply %2599, %2600 : tensor<1x32x64xcomplex<f32>>
    %2602 = stablehlo.real %2601 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2603 = stablehlo.reshape %2602 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2604 = stablehlo.imag %2601 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2605 = stablehlo.reshape %2604 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2606 = stablehlo.concatenate %2603, %2605, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2607 = stablehlo.reshape %2606 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %2608 = "stablehlo.scatter"(%arg329, %2591, %2607) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2609 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2610 = "stablehlo.gather"(%2608, %2609) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2611 = stablehlo.transpose %2610, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %2612 = stablehlo.dot_general %2586, %2611, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %2613 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %2614 = stablehlo.divide %2612, %2613 : tensor<32x1x2049xf32>
    %2615 = stablehlo.reduce(%2614 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2616 = stablehlo.broadcast_in_dim %2615, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2617 = stablehlo.subtract %2614, %2616 : tensor<32x1x2049xf32>
    %2618 = stablehlo.exponential %2617 : tensor<32x1x2049xf32>
    %2619 = stablehlo.reduce(%2618 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2620 = stablehlo.broadcast_in_dim %2619, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2621 = stablehlo.divide %2618, %2620 : tensor<32x1x2049xf32>
    %2622 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2623 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2624 = stablehlo.add %arg198, %2623 : tensor<1xi64>
    %2625 = stablehlo.select %2622, %2624, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2626 = stablehlo.reshape %2625 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2627 = stablehlo.transpose %arg44, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2628 = stablehlo.dot %2570, %2627, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2629 = stablehlo.reshape %2628 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %2630 = "stablehlo.scatter"(%arg327, %2626, %2629) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2631 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2632 = "stablehlo.gather"(%2630, %2631) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2633 = stablehlo.transpose %2632, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %2634 = stablehlo.dot_general %2621, %2633, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %2635 = stablehlo.reshape %2634 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %2636 = stablehlo.transpose %arg43, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2637 = stablehlo.dot %2635, %2636, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2638 = stablehlo.add %2558, %2637 : tensor<1x4096xf32>
    %2639 = stablehlo.power %2638, %1 : tensor<1x4096xf32>
    %2640 = stablehlo.reduce(%2639 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2641 = stablehlo.multiply %2640, %0 : tensor<1xf32>
    %2642 = stablehlo.reshape %2641 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2643 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2644 = stablehlo.add %2642, %2643 : tensor<1x1xf32>
    %2645 = stablehlo.rsqrt %2644 : tensor<1x1xf32>
    %2646 = stablehlo.reshape %2645 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2647 = stablehlo.broadcast_in_dim %2646, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2648 = stablehlo.multiply %2638, %2647 : tensor<1x4096xf32>
    %2649 = stablehlo.reshape %arg42 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2650 = stablehlo.multiply %2648, %2649 : tensor<1x4096xf32>
    %2651 = stablehlo.transpose %arg331, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2652 = stablehlo.dot %2650, %2651, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2653 = stablehlo.logistic %2652 : tensor<1x11008xf32>
    %2654 = stablehlo.multiply %2652, %2653 : tensor<1x11008xf32>
    %2655 = stablehlo.transpose %arg41, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2656 = stablehlo.dot %2650, %2655, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2657 = stablehlo.multiply %2654, %2656 : tensor<1x11008xf32>
    %2658 = stablehlo.transpose %arg40, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2659 = stablehlo.dot %2657, %2658, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %2660 = stablehlo.add %2638, %2659 : tensor<1x4096xf32>
    %2661 = stablehlo.power %2660, %1 : tensor<1x4096xf32>
    %2662 = stablehlo.reduce(%2661 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2663 = stablehlo.multiply %2662, %0 : tensor<1xf32>
    %2664 = stablehlo.reshape %2663 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2665 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2666 = stablehlo.add %2664, %2665 : tensor<1x1xf32>
    %2667 = stablehlo.rsqrt %2666 : tensor<1x1xf32>
    %2668 = stablehlo.reshape %2667 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2669 = stablehlo.broadcast_in_dim %2668, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2670 = stablehlo.multiply %2660, %2669 : tensor<1x4096xf32>
    %2671 = stablehlo.reshape %arg39 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2672 = stablehlo.multiply %2670, %2671 : tensor<1x4096xf32>
    %2673 = stablehlo.transpose %arg335, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2674 = stablehlo.dot %2672, %2673, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2675 = stablehlo.reshape %2674 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2676 = stablehlo.slice %2675 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2677 = stablehlo.reshape %2676 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2678 = stablehlo.slice %2675 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2679 = stablehlo.reshape %2678 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2680 = stablehlo.complex %2677, %2679 : tensor<1x32x64xcomplex<f32>>
    %2681 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2682 = stablehlo.multiply %2680, %2681 : tensor<1x32x64xcomplex<f32>>
    %2683 = stablehlo.real %2682 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2684 = stablehlo.reshape %2683 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2685 = stablehlo.imag %2682 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2686 = stablehlo.reshape %2685 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2687 = stablehlo.concatenate %2684, %2686, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2688 = stablehlo.reshape %2687 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %2689 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2690 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2691 = stablehlo.add %arg198, %2690 : tensor<1xi64>
    %2692 = stablehlo.select %2689, %2691, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2693 = stablehlo.reshape %2692 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2694 = stablehlo.transpose %arg333, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2695 = stablehlo.dot %2672, %2694, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2696 = stablehlo.reshape %2695 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2697 = stablehlo.slice %2696 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2698 = stablehlo.reshape %2697 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2699 = stablehlo.slice %2696 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2700 = stablehlo.reshape %2699 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2701 = stablehlo.complex %2698, %2700 : tensor<1x32x64xcomplex<f32>>
    %2702 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2703 = stablehlo.multiply %2701, %2702 : tensor<1x32x64xcomplex<f32>>
    %2704 = stablehlo.real %2703 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2705 = stablehlo.reshape %2704 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2706 = stablehlo.imag %2703 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2707 = stablehlo.reshape %2706 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2708 = stablehlo.concatenate %2705, %2707, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2709 = stablehlo.reshape %2708 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %2710 = "stablehlo.scatter"(%arg334, %2693, %2709) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2711 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2712 = "stablehlo.gather"(%2710, %2711) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2713 = stablehlo.transpose %2712, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %2714 = stablehlo.dot_general %2688, %2713, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %2715 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %2716 = stablehlo.divide %2714, %2715 : tensor<32x1x2049xf32>
    %2717 = stablehlo.reduce(%2716 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2718 = stablehlo.broadcast_in_dim %2717, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2719 = stablehlo.subtract %2716, %2718 : tensor<32x1x2049xf32>
    %2720 = stablehlo.exponential %2719 : tensor<32x1x2049xf32>
    %2721 = stablehlo.reduce(%2720 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2722 = stablehlo.broadcast_in_dim %2721, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2723 = stablehlo.divide %2720, %2722 : tensor<32x1x2049xf32>
    %2724 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2725 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2726 = stablehlo.add %arg198, %2725 : tensor<1xi64>
    %2727 = stablehlo.select %2724, %2726, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2728 = stablehlo.reshape %2727 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2729 = stablehlo.transpose %arg38, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2730 = stablehlo.dot %2672, %2729, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2731 = stablehlo.reshape %2730 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %2732 = "stablehlo.scatter"(%arg332, %2728, %2731) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2733 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2734 = "stablehlo.gather"(%2732, %2733) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2735 = stablehlo.transpose %2734, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %2736 = stablehlo.dot_general %2723, %2735, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %2737 = stablehlo.reshape %2736 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %2738 = stablehlo.transpose %arg37, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2739 = stablehlo.dot %2737, %2738, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2740 = stablehlo.add %2660, %2739 : tensor<1x4096xf32>
    %2741 = stablehlo.power %2740, %1 : tensor<1x4096xf32>
    %2742 = stablehlo.reduce(%2741 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2743 = stablehlo.multiply %2742, %0 : tensor<1xf32>
    %2744 = stablehlo.reshape %2743 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2745 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2746 = stablehlo.add %2744, %2745 : tensor<1x1xf32>
    %2747 = stablehlo.rsqrt %2746 : tensor<1x1xf32>
    %2748 = stablehlo.reshape %2747 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2749 = stablehlo.broadcast_in_dim %2748, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2750 = stablehlo.multiply %2740, %2749 : tensor<1x4096xf32>
    %2751 = stablehlo.reshape %arg36 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2752 = stablehlo.multiply %2750, %2751 : tensor<1x4096xf32>
    %2753 = stablehlo.transpose %arg336, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2754 = stablehlo.dot %2752, %2753, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2755 = stablehlo.logistic %2754 : tensor<1x11008xf32>
    %2756 = stablehlo.multiply %2754, %2755 : tensor<1x11008xf32>
    %2757 = stablehlo.transpose %arg35, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2758 = stablehlo.dot %2752, %2757, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2759 = stablehlo.multiply %2756, %2758 : tensor<1x11008xf32>
    %2760 = stablehlo.transpose %arg34, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2761 = stablehlo.dot %2759, %2760, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %2762 = stablehlo.add %2740, %2761 : tensor<1x4096xf32>
    %2763 = stablehlo.power %2762, %1 : tensor<1x4096xf32>
    %2764 = stablehlo.reduce(%2763 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2765 = stablehlo.multiply %2764, %0 : tensor<1xf32>
    %2766 = stablehlo.reshape %2765 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2767 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2768 = stablehlo.add %2766, %2767 : tensor<1x1xf32>
    %2769 = stablehlo.rsqrt %2768 : tensor<1x1xf32>
    %2770 = stablehlo.reshape %2769 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2771 = stablehlo.broadcast_in_dim %2770, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2772 = stablehlo.multiply %2762, %2771 : tensor<1x4096xf32>
    %2773 = stablehlo.reshape %arg33 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2774 = stablehlo.multiply %2772, %2773 : tensor<1x4096xf32>
    %2775 = stablehlo.transpose %arg340, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2776 = stablehlo.dot %2774, %2775, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2777 = stablehlo.reshape %2776 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2778 = stablehlo.slice %2777 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2779 = stablehlo.reshape %2778 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2780 = stablehlo.slice %2777 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2781 = stablehlo.reshape %2780 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2782 = stablehlo.complex %2779, %2781 : tensor<1x32x64xcomplex<f32>>
    %2783 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2784 = stablehlo.multiply %2782, %2783 : tensor<1x32x64xcomplex<f32>>
    %2785 = stablehlo.real %2784 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2786 = stablehlo.reshape %2785 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2787 = stablehlo.imag %2784 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2788 = stablehlo.reshape %2787 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2789 = stablehlo.concatenate %2786, %2788, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2790 = stablehlo.reshape %2789 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %2791 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2792 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2793 = stablehlo.add %arg198, %2792 : tensor<1xi64>
    %2794 = stablehlo.select %2791, %2793, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2795 = stablehlo.reshape %2794 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2796 = stablehlo.transpose %arg338, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2797 = stablehlo.dot %2774, %2796, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2798 = stablehlo.reshape %2797 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2799 = stablehlo.slice %2798 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2800 = stablehlo.reshape %2799 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2801 = stablehlo.slice %2798 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2802 = stablehlo.reshape %2801 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2803 = stablehlo.complex %2800, %2802 : tensor<1x32x64xcomplex<f32>>
    %2804 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2805 = stablehlo.multiply %2803, %2804 : tensor<1x32x64xcomplex<f32>>
    %2806 = stablehlo.real %2805 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2807 = stablehlo.reshape %2806 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2808 = stablehlo.imag %2805 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2809 = stablehlo.reshape %2808 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2810 = stablehlo.concatenate %2807, %2809, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2811 = stablehlo.reshape %2810 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %2812 = "stablehlo.scatter"(%arg339, %2795, %2811) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2813 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2814 = "stablehlo.gather"(%2812, %2813) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2815 = stablehlo.transpose %2814, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %2816 = stablehlo.dot_general %2790, %2815, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %2817 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %2818 = stablehlo.divide %2816, %2817 : tensor<32x1x2049xf32>
    %2819 = stablehlo.reduce(%2818 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2820 = stablehlo.broadcast_in_dim %2819, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2821 = stablehlo.subtract %2818, %2820 : tensor<32x1x2049xf32>
    %2822 = stablehlo.exponential %2821 : tensor<32x1x2049xf32>
    %2823 = stablehlo.reduce(%2822 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2824 = stablehlo.broadcast_in_dim %2823, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2825 = stablehlo.divide %2822, %2824 : tensor<32x1x2049xf32>
    %2826 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2827 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2828 = stablehlo.add %arg198, %2827 : tensor<1xi64>
    %2829 = stablehlo.select %2826, %2828, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2830 = stablehlo.reshape %2829 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2831 = stablehlo.transpose %arg32, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2832 = stablehlo.dot %2774, %2831, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2833 = stablehlo.reshape %2832 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %2834 = "stablehlo.scatter"(%arg337, %2830, %2833) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2835 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2836 = "stablehlo.gather"(%2834, %2835) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2837 = stablehlo.transpose %2836, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %2838 = stablehlo.dot_general %2825, %2837, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %2839 = stablehlo.reshape %2838 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %2840 = stablehlo.transpose %arg31, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2841 = stablehlo.dot %2839, %2840, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2842 = stablehlo.add %2762, %2841 : tensor<1x4096xf32>
    %2843 = stablehlo.power %2842, %1 : tensor<1x4096xf32>
    %2844 = stablehlo.reduce(%2843 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2845 = stablehlo.multiply %2844, %0 : tensor<1xf32>
    %2846 = stablehlo.reshape %2845 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2847 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2848 = stablehlo.add %2846, %2847 : tensor<1x1xf32>
    %2849 = stablehlo.rsqrt %2848 : tensor<1x1xf32>
    %2850 = stablehlo.reshape %2849 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2851 = stablehlo.broadcast_in_dim %2850, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2852 = stablehlo.multiply %2842, %2851 : tensor<1x4096xf32>
    %2853 = stablehlo.reshape %arg30 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2854 = stablehlo.multiply %2852, %2853 : tensor<1x4096xf32>
    %2855 = stablehlo.transpose %arg341, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2856 = stablehlo.dot %2854, %2855, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2857 = stablehlo.logistic %2856 : tensor<1x11008xf32>
    %2858 = stablehlo.multiply %2856, %2857 : tensor<1x11008xf32>
    %2859 = stablehlo.transpose %arg29, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2860 = stablehlo.dot %2854, %2859, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2861 = stablehlo.multiply %2858, %2860 : tensor<1x11008xf32>
    %2862 = stablehlo.transpose %arg28, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2863 = stablehlo.dot %2861, %2862, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %2864 = stablehlo.add %2842, %2863 : tensor<1x4096xf32>
    %2865 = stablehlo.power %2864, %1 : tensor<1x4096xf32>
    %2866 = stablehlo.reduce(%2865 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2867 = stablehlo.multiply %2866, %0 : tensor<1xf32>
    %2868 = stablehlo.reshape %2867 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2869 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2870 = stablehlo.add %2868, %2869 : tensor<1x1xf32>
    %2871 = stablehlo.rsqrt %2870 : tensor<1x1xf32>
    %2872 = stablehlo.reshape %2871 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2873 = stablehlo.broadcast_in_dim %2872, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2874 = stablehlo.multiply %2864, %2873 : tensor<1x4096xf32>
    %2875 = stablehlo.reshape %arg27 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2876 = stablehlo.multiply %2874, %2875 : tensor<1x4096xf32>
    %2877 = stablehlo.transpose %arg345, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2878 = stablehlo.dot %2876, %2877, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2879 = stablehlo.reshape %2878 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2880 = stablehlo.slice %2879 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2881 = stablehlo.reshape %2880 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2882 = stablehlo.slice %2879 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2883 = stablehlo.reshape %2882 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2884 = stablehlo.complex %2881, %2883 : tensor<1x32x64xcomplex<f32>>
    %2885 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2886 = stablehlo.multiply %2884, %2885 : tensor<1x32x64xcomplex<f32>>
    %2887 = stablehlo.real %2886 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2888 = stablehlo.reshape %2887 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2889 = stablehlo.imag %2886 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2890 = stablehlo.reshape %2889 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2891 = stablehlo.concatenate %2888, %2890, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2892 = stablehlo.reshape %2891 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %2893 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2894 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2895 = stablehlo.add %arg198, %2894 : tensor<1xi64>
    %2896 = stablehlo.select %2893, %2895, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2897 = stablehlo.reshape %2896 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2898 = stablehlo.transpose %arg343, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2899 = stablehlo.dot %2876, %2898, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2900 = stablehlo.reshape %2899 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2901 = stablehlo.slice %2900 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2902 = stablehlo.reshape %2901 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2903 = stablehlo.slice %2900 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2904 = stablehlo.reshape %2903 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2905 = stablehlo.complex %2902, %2904 : tensor<1x32x64xcomplex<f32>>
    %2906 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2907 = stablehlo.multiply %2905, %2906 : tensor<1x32x64xcomplex<f32>>
    %2908 = stablehlo.real %2907 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2909 = stablehlo.reshape %2908 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2910 = stablehlo.imag %2907 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2911 = stablehlo.reshape %2910 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2912 = stablehlo.concatenate %2909, %2911, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2913 = stablehlo.reshape %2912 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %2914 = "stablehlo.scatter"(%arg344, %2897, %2913) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2915 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2916 = "stablehlo.gather"(%2914, %2915) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2917 = stablehlo.transpose %2916, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %2918 = stablehlo.dot_general %2892, %2917, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %2919 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %2920 = stablehlo.divide %2918, %2919 : tensor<32x1x2049xf32>
    %2921 = stablehlo.reduce(%2920 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2922 = stablehlo.broadcast_in_dim %2921, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2923 = stablehlo.subtract %2920, %2922 : tensor<32x1x2049xf32>
    %2924 = stablehlo.exponential %2923 : tensor<32x1x2049xf32>
    %2925 = stablehlo.reduce(%2924 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2926 = stablehlo.broadcast_in_dim %2925, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %2927 = stablehlo.divide %2924, %2926 : tensor<32x1x2049xf32>
    %2928 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2929 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2930 = stablehlo.add %arg198, %2929 : tensor<1xi64>
    %2931 = stablehlo.select %2928, %2930, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2932 = stablehlo.reshape %2931 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2933 = stablehlo.transpose %arg26, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2934 = stablehlo.dot %2876, %2933, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2935 = stablehlo.reshape %2934 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %2936 = "stablehlo.scatter"(%arg342, %2932, %2935) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %2937 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %2938 = "stablehlo.gather"(%2936, %2937) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %2939 = stablehlo.transpose %2938, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %2940 = stablehlo.dot_general %2927, %2939, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %2941 = stablehlo.reshape %2940 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %2942 = stablehlo.transpose %arg25, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2943 = stablehlo.dot %2941, %2942, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2944 = stablehlo.add %2864, %2943 : tensor<1x4096xf32>
    %2945 = stablehlo.power %2944, %1 : tensor<1x4096xf32>
    %2946 = stablehlo.reduce(%2945 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2947 = stablehlo.multiply %2946, %0 : tensor<1xf32>
    %2948 = stablehlo.reshape %2947 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2949 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2950 = stablehlo.add %2948, %2949 : tensor<1x1xf32>
    %2951 = stablehlo.rsqrt %2950 : tensor<1x1xf32>
    %2952 = stablehlo.reshape %2951 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2953 = stablehlo.broadcast_in_dim %2952, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2954 = stablehlo.multiply %2944, %2953 : tensor<1x4096xf32>
    %2955 = stablehlo.reshape %arg24 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2956 = stablehlo.multiply %2954, %2955 : tensor<1x4096xf32>
    %2957 = stablehlo.transpose %arg346, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2958 = stablehlo.dot %2956, %2957, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2959 = stablehlo.logistic %2958 : tensor<1x11008xf32>
    %2960 = stablehlo.multiply %2958, %2959 : tensor<1x11008xf32>
    %2961 = stablehlo.transpose %arg23, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2962 = stablehlo.dot %2956, %2961, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %2963 = stablehlo.multiply %2960, %2962 : tensor<1x11008xf32>
    %2964 = stablehlo.transpose %arg22, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2965 = stablehlo.dot %2963, %2964, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %2966 = stablehlo.add %2944, %2965 : tensor<1x4096xf32>
    %2967 = stablehlo.power %2966, %1 : tensor<1x4096xf32>
    %2968 = stablehlo.reduce(%2967 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %2969 = stablehlo.multiply %2968, %0 : tensor<1xf32>
    %2970 = stablehlo.reshape %2969 : (tensor<1xf32>) -> tensor<1x1xf32>
    %2971 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %2972 = stablehlo.add %2970, %2971 : tensor<1x1xf32>
    %2973 = stablehlo.rsqrt %2972 : tensor<1x1xf32>
    %2974 = stablehlo.reshape %2973 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2975 = stablehlo.broadcast_in_dim %2974, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %2976 = stablehlo.multiply %2966, %2975 : tensor<1x4096xf32>
    %2977 = stablehlo.reshape %arg21 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %2978 = stablehlo.multiply %2976, %2977 : tensor<1x4096xf32>
    %2979 = stablehlo.transpose %arg350, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2980 = stablehlo.dot %2978, %2979, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %2981 = stablehlo.reshape %2980 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %2982 = stablehlo.slice %2981 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2983 = stablehlo.reshape %2982 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2984 = stablehlo.slice %2981 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %2985 = stablehlo.reshape %2984 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %2986 = stablehlo.complex %2983, %2985 : tensor<1x32x64xcomplex<f32>>
    %2987 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %2988 = stablehlo.multiply %2986, %2987 : tensor<1x32x64xcomplex<f32>>
    %2989 = stablehlo.real %2988 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2990 = stablehlo.reshape %2989 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2991 = stablehlo.imag %2988 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %2992 = stablehlo.reshape %2991 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %2993 = stablehlo.concatenate %2990, %2992, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %2994 = stablehlo.reshape %2993 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %2995 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2996 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %2997 = stablehlo.add %arg198, %2996 : tensor<1xi64>
    %2998 = stablehlo.select %2995, %2997, %arg198 : tensor<1xi1>, tensor<1xi64>
    %2999 = stablehlo.reshape %2998 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3000 = stablehlo.transpose %arg348, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3001 = stablehlo.dot %2978, %3000, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %3002 = stablehlo.reshape %3001 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %3003 = stablehlo.slice %3002 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %3004 = stablehlo.reshape %3003 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3005 = stablehlo.slice %3002 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %3006 = stablehlo.reshape %3005 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3007 = stablehlo.complex %3004, %3006 : tensor<1x32x64xcomplex<f32>>
    %3008 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %3009 = stablehlo.multiply %3007, %3008 : tensor<1x32x64xcomplex<f32>>
    %3010 = stablehlo.real %3009 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %3011 = stablehlo.reshape %3010 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %3012 = stablehlo.imag %3009 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %3013 = stablehlo.reshape %3012 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %3014 = stablehlo.concatenate %3011, %3013, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %3015 = stablehlo.reshape %3014 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %3016 = "stablehlo.scatter"(%arg349, %2999, %3015) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %3017 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %3018 = "stablehlo.gather"(%3016, %3017) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %3019 = stablehlo.transpose %3018, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %3020 = stablehlo.dot_general %2994, %3019, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %3021 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %3022 = stablehlo.divide %3020, %3021 : tensor<32x1x2049xf32>
    %3023 = stablehlo.reduce(%3022 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %3024 = stablehlo.broadcast_in_dim %3023, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %3025 = stablehlo.subtract %3022, %3024 : tensor<32x1x2049xf32>
    %3026 = stablehlo.exponential %3025 : tensor<32x1x2049xf32>
    %3027 = stablehlo.reduce(%3026 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %3028 = stablehlo.broadcast_in_dim %3027, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %3029 = stablehlo.divide %3026, %3028 : tensor<32x1x2049xf32>
    %3030 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3031 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %3032 = stablehlo.add %arg198, %3031 : tensor<1xi64>
    %3033 = stablehlo.select %3030, %3032, %arg198 : tensor<1xi1>, tensor<1xi64>
    %3034 = stablehlo.reshape %3033 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3035 = stablehlo.transpose %arg20, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3036 = stablehlo.dot %2978, %3035, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %3037 = stablehlo.reshape %3036 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %3038 = "stablehlo.scatter"(%arg347, %3034, %3037) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %3039 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %3040 = "stablehlo.gather"(%3038, %3039) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %3041 = stablehlo.transpose %3040, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %3042 = stablehlo.dot_general %3029, %3041, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %3043 = stablehlo.reshape %3042 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %3044 = stablehlo.transpose %arg19, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3045 = stablehlo.dot %3043, %3044, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %3046 = stablehlo.add %2966, %3045 : tensor<1x4096xf32>
    %3047 = stablehlo.power %3046, %1 : tensor<1x4096xf32>
    %3048 = stablehlo.reduce(%3047 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %3049 = stablehlo.multiply %3048, %0 : tensor<1xf32>
    %3050 = stablehlo.reshape %3049 : (tensor<1xf32>) -> tensor<1x1xf32>
    %3051 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %3052 = stablehlo.add %3050, %3051 : tensor<1x1xf32>
    %3053 = stablehlo.rsqrt %3052 : tensor<1x1xf32>
    %3054 = stablehlo.reshape %3053 : (tensor<1x1xf32>) -> tensor<1xf32>
    %3055 = stablehlo.broadcast_in_dim %3054, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %3056 = stablehlo.multiply %3046, %3055 : tensor<1x4096xf32>
    %3057 = stablehlo.reshape %arg18 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %3058 = stablehlo.multiply %3056, %3057 : tensor<1x4096xf32>
    %3059 = stablehlo.transpose %arg351, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3060 = stablehlo.dot %3058, %3059, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %3061 = stablehlo.logistic %3060 : tensor<1x11008xf32>
    %3062 = stablehlo.multiply %3060, %3061 : tensor<1x11008xf32>
    %3063 = stablehlo.transpose %arg17, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3064 = stablehlo.dot %3058, %3063, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %3065 = stablehlo.multiply %3062, %3064 : tensor<1x11008xf32>
    %3066 = stablehlo.transpose %arg16, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %3067 = stablehlo.dot %3065, %3066, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %3068 = stablehlo.add %3046, %3067 : tensor<1x4096xf32>
    %3069 = stablehlo.power %3068, %1 : tensor<1x4096xf32>
    %3070 = stablehlo.reduce(%3069 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %3071 = stablehlo.multiply %3070, %0 : tensor<1xf32>
    %3072 = stablehlo.reshape %3071 : (tensor<1xf32>) -> tensor<1x1xf32>
    %3073 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %3074 = stablehlo.add %3072, %3073 : tensor<1x1xf32>
    %3075 = stablehlo.rsqrt %3074 : tensor<1x1xf32>
    %3076 = stablehlo.reshape %3075 : (tensor<1x1xf32>) -> tensor<1xf32>
    %3077 = stablehlo.broadcast_in_dim %3076, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %3078 = stablehlo.multiply %3068, %3077 : tensor<1x4096xf32>
    %3079 = stablehlo.reshape %arg15 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %3080 = stablehlo.multiply %3078, %3079 : tensor<1x4096xf32>
    %3081 = stablehlo.transpose %arg355, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3082 = stablehlo.dot %3080, %3081, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %3083 = stablehlo.reshape %3082 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %3084 = stablehlo.slice %3083 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %3085 = stablehlo.reshape %3084 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3086 = stablehlo.slice %3083 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %3087 = stablehlo.reshape %3086 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3088 = stablehlo.complex %3085, %3087 : tensor<1x32x64xcomplex<f32>>
    %3089 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %3090 = stablehlo.multiply %3088, %3089 : tensor<1x32x64xcomplex<f32>>
    %3091 = stablehlo.real %3090 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %3092 = stablehlo.reshape %3091 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %3093 = stablehlo.imag %3090 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %3094 = stablehlo.reshape %3093 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %3095 = stablehlo.concatenate %3092, %3094, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %3096 = stablehlo.reshape %3095 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %3097 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3098 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %3099 = stablehlo.add %arg198, %3098 : tensor<1xi64>
    %3100 = stablehlo.select %3097, %3099, %arg198 : tensor<1xi1>, tensor<1xi64>
    %3101 = stablehlo.reshape %3100 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3102 = stablehlo.transpose %arg353, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3103 = stablehlo.dot %3080, %3102, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %3104 = stablehlo.reshape %3103 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %3105 = stablehlo.slice %3104 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %3106 = stablehlo.reshape %3105 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3107 = stablehlo.slice %3104 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %3108 = stablehlo.reshape %3107 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3109 = stablehlo.complex %3106, %3108 : tensor<1x32x64xcomplex<f32>>
    %3110 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %3111 = stablehlo.multiply %3109, %3110 : tensor<1x32x64xcomplex<f32>>
    %3112 = stablehlo.real %3111 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %3113 = stablehlo.reshape %3112 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %3114 = stablehlo.imag %3111 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %3115 = stablehlo.reshape %3114 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %3116 = stablehlo.concatenate %3113, %3115, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %3117 = stablehlo.reshape %3116 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %3118 = "stablehlo.scatter"(%arg354, %3101, %3117) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %3119 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %3120 = "stablehlo.gather"(%3118, %3119) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %3121 = stablehlo.transpose %3120, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %3122 = stablehlo.dot_general %3096, %3121, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %3123 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %3124 = stablehlo.divide %3122, %3123 : tensor<32x1x2049xf32>
    %3125 = stablehlo.reduce(%3124 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %3126 = stablehlo.broadcast_in_dim %3125, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %3127 = stablehlo.subtract %3124, %3126 : tensor<32x1x2049xf32>
    %3128 = stablehlo.exponential %3127 : tensor<32x1x2049xf32>
    %3129 = stablehlo.reduce(%3128 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %3130 = stablehlo.broadcast_in_dim %3129, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %3131 = stablehlo.divide %3128, %3130 : tensor<32x1x2049xf32>
    %3132 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3133 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %3134 = stablehlo.add %arg198, %3133 : tensor<1xi64>
    %3135 = stablehlo.select %3132, %3134, %arg198 : tensor<1xi1>, tensor<1xi64>
    %3136 = stablehlo.reshape %3135 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3137 = stablehlo.transpose %arg14, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3138 = stablehlo.dot %3080, %3137, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %3139 = stablehlo.reshape %3138 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %3140 = "stablehlo.scatter"(%arg352, %3136, %3139) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %3141 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %3142 = "stablehlo.gather"(%3140, %3141) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %3143 = stablehlo.transpose %3142, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %3144 = stablehlo.dot_general %3131, %3143, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %3145 = stablehlo.reshape %3144 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %3146 = stablehlo.transpose %arg13, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3147 = stablehlo.dot %3145, %3146, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %3148 = stablehlo.add %3068, %3147 : tensor<1x4096xf32>
    %3149 = stablehlo.power %3148, %1 : tensor<1x4096xf32>
    %3150 = stablehlo.reduce(%3149 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %3151 = stablehlo.multiply %3150, %0 : tensor<1xf32>
    %3152 = stablehlo.reshape %3151 : (tensor<1xf32>) -> tensor<1x1xf32>
    %3153 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %3154 = stablehlo.add %3152, %3153 : tensor<1x1xf32>
    %3155 = stablehlo.rsqrt %3154 : tensor<1x1xf32>
    %3156 = stablehlo.reshape %3155 : (tensor<1x1xf32>) -> tensor<1xf32>
    %3157 = stablehlo.broadcast_in_dim %3156, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %3158 = stablehlo.multiply %3148, %3157 : tensor<1x4096xf32>
    %3159 = stablehlo.reshape %arg12 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %3160 = stablehlo.multiply %3158, %3159 : tensor<1x4096xf32>
    %3161 = stablehlo.transpose %arg356, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3162 = stablehlo.dot %3160, %3161, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %3163 = stablehlo.logistic %3162 : tensor<1x11008xf32>
    %3164 = stablehlo.multiply %3162, %3163 : tensor<1x11008xf32>
    %3165 = stablehlo.transpose %arg11, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3166 = stablehlo.dot %3160, %3165, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %3167 = stablehlo.multiply %3164, %3166 : tensor<1x11008xf32>
    %3168 = stablehlo.transpose %arg10, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %3169 = stablehlo.dot %3167, %3168, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %3170 = stablehlo.add %3148, %3169 : tensor<1x4096xf32>
    %3171 = stablehlo.power %3170, %1 : tensor<1x4096xf32>
    %3172 = stablehlo.reduce(%3171 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %3173 = stablehlo.multiply %3172, %0 : tensor<1xf32>
    %3174 = stablehlo.reshape %3173 : (tensor<1xf32>) -> tensor<1x1xf32>
    %3175 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %3176 = stablehlo.add %3174, %3175 : tensor<1x1xf32>
    %3177 = stablehlo.rsqrt %3176 : tensor<1x1xf32>
    %3178 = stablehlo.reshape %3177 : (tensor<1x1xf32>) -> tensor<1xf32>
    %3179 = stablehlo.broadcast_in_dim %3178, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %3180 = stablehlo.multiply %3170, %3179 : tensor<1x4096xf32>
    %3181 = stablehlo.reshape %arg9 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %3182 = stablehlo.multiply %3180, %3181 : tensor<1x4096xf32>
    %3183 = stablehlo.transpose %arg360, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3184 = stablehlo.dot %3182, %3183, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %3185 = stablehlo.reshape %3184 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %3186 = stablehlo.slice %3185 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %3187 = stablehlo.reshape %3186 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3188 = stablehlo.slice %3185 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %3189 = stablehlo.reshape %3188 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3190 = stablehlo.complex %3187, %3189 : tensor<1x32x64xcomplex<f32>>
    %3191 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %3192 = stablehlo.multiply %3190, %3191 : tensor<1x32x64xcomplex<f32>>
    %3193 = stablehlo.real %3192 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %3194 = stablehlo.reshape %3193 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %3195 = stablehlo.imag %3192 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %3196 = stablehlo.reshape %3195 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %3197 = stablehlo.concatenate %3194, %3196, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %3198 = stablehlo.reshape %3197 : (tensor<1x32x64x2xf32>) -> tensor<32x1x128xf32>
    %3199 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3200 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %3201 = stablehlo.add %arg198, %3200 : tensor<1xi64>
    %3202 = stablehlo.select %3199, %3201, %arg198 : tensor<1xi1>, tensor<1xi64>
    %3203 = stablehlo.reshape %3202 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3204 = stablehlo.transpose %arg358, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3205 = stablehlo.dot %3182, %3204, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %3206 = stablehlo.reshape %3205 : (tensor<1x4096xf32>) -> tensor<1x32x64x2xf32>
    %3207 = stablehlo.slice %3206 [0:1, 0:32, 0:64, 0:1] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %3208 = stablehlo.reshape %3207 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3209 = stablehlo.slice %3206 [0:1, 0:32, 0:64, 1:2] : (tensor<1x32x64x2xf32>) -> tensor<1x32x64x1xf32>
    %3210 = stablehlo.reshape %3209 : (tensor<1x32x64x1xf32>) -> tensor<1x32x64xf32>
    %3211 = stablehlo.complex %3208, %3210 : tensor<1x32x64xcomplex<f32>>
    %3212 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x64xcomplex<f32>>) -> tensor<1x32x64xcomplex<f32>>
    %3213 = stablehlo.multiply %3211, %3212 : tensor<1x32x64xcomplex<f32>>
    %3214 = stablehlo.real %3213 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %3215 = stablehlo.reshape %3214 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %3216 = stablehlo.imag %3213 : (tensor<1x32x64xcomplex<f32>>) -> tensor<1x32x64xf32>
    %3217 = stablehlo.reshape %3216 : (tensor<1x32x64xf32>) -> tensor<1x32x64x1xf32>
    %3218 = stablehlo.concatenate %3215, %3217, dim = 3 : (tensor<1x32x64x1xf32>, tensor<1x32x64x1xf32>) -> tensor<1x32x64x2xf32>
    %3219 = stablehlo.reshape %3218 : (tensor<1x32x64x2xf32>) -> tensor<1x32x128xf32>
    %3220 = "stablehlo.scatter"(%arg359, %3203, %3219) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %3221 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %3222 = "stablehlo.gather"(%3220, %3221) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %3223 = stablehlo.transpose %3222, dims = [1, 2, 0] : (tensor<2049x32x128xf32>) -> tensor<32x128x2049xf32>
    %3224 = stablehlo.dot_general %3198, %3223, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x128xf32>, tensor<32x128x2049xf32>) -> tensor<32x1x2049xf32>
    %3225 = stablehlo.broadcast_in_dim %arg201, dims = [] : (tensor<f32>) -> tensor<32x1x2049xf32>
    %3226 = stablehlo.divide %3224, %3225 : tensor<32x1x2049xf32>
    %3227 = stablehlo.reduce(%3226 init: %3) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.maximum %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %3228 = stablehlo.broadcast_in_dim %3227, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %3229 = stablehlo.subtract %3226, %3228 : tensor<32x1x2049xf32>
    %3230 = stablehlo.exponential %3229 : tensor<32x1x2049xf32>
    %3231 = stablehlo.reduce(%3230 init: %4) across dimensions = [2] : (tensor<32x1x2049xf32>, tensor<f32>) -> tensor<32x1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %3232 = stablehlo.broadcast_in_dim %3231, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x2049xf32>
    %3233 = stablehlo.divide %3230, %3232 : tensor<32x1x2049xf32>
    %3234 = stablehlo.compare  LT, %arg198, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3235 = stablehlo.reshape %arg199 : (tensor<i64>) -> tensor<1xi64>
    %3236 = stablehlo.add %arg198, %3235 : tensor<1xi64>
    %3237 = stablehlo.select %3234, %3236, %arg198 : tensor<1xi1>, tensor<1xi64>
    %3238 = stablehlo.reshape %3237 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3239 = stablehlo.transpose %arg8, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3240 = stablehlo.dot %3182, %3239, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %3241 = stablehlo.reshape %3240 : (tensor<1x4096xf32>) -> tensor<1x32x128xf32>
    %3242 = "stablehlo.scatter"(%arg357, %3238, %3241) ({
    ^bb0(%arg362: tensor<f32>, %arg363: tensor<f32>):
      stablehlo.return %arg363 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x32x128xf32>, tensor<1x1xi64>, tensor<1x32x128xf32>) -> tensor<2304x32x128xf32>
    %3243 = stablehlo.convert %arg7 : (tensor<2049xi64>) -> tensor<2049xui32>
    %3244 = "stablehlo.gather"(%3242, %3243) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32, 128]> : tensor<3xi64>} : (tensor<2304x32x128xf32>, tensor<2049xui32>) -> tensor<2049x32x128xf32>
    %3245 = stablehlo.transpose %3244, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[32,2049,128]{2,0,1}"} : (tensor<2049x32x128xf32>) -> tensor<32x2049x128xf32>
    %3246 = stablehlo.dot_general %3233, %3245, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x2049xf32>, tensor<32x2049x128xf32>) -> tensor<32x1x128xf32>
    %3247 = stablehlo.reshape %3246 : (tensor<32x1x128xf32>) -> tensor<1x4096xf32>
    %3248 = stablehlo.transpose %arg6, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3249 = stablehlo.dot %3247, %3248, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x4096xf32>
    %3250 = stablehlo.add %3170, %3249 : tensor<1x4096xf32>
    %3251 = stablehlo.power %3250, %1 : tensor<1x4096xf32>
    %3252 = stablehlo.reduce(%3251 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %3253 = stablehlo.multiply %3252, %0 : tensor<1xf32>
    %3254 = stablehlo.reshape %3253 : (tensor<1xf32>) -> tensor<1x1xf32>
    %3255 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %3256 = stablehlo.add %3254, %3255 : tensor<1x1xf32>
    %3257 = stablehlo.rsqrt %3256 : tensor<1x1xf32>
    %3258 = stablehlo.reshape %3257 : (tensor<1x1xf32>) -> tensor<1xf32>
    %3259 = stablehlo.broadcast_in_dim %3258, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %3260 = stablehlo.multiply %3250, %3259 : tensor<1x4096xf32>
    %3261 = stablehlo.reshape %arg5 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %3262 = stablehlo.multiply %3260, %3261 : tensor<1x4096xf32>
    %3263 = stablehlo.transpose %arg361, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3264 = stablehlo.dot %3262, %3263, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %3265 = stablehlo.logistic %3264 : tensor<1x11008xf32>
    %3266 = stablehlo.multiply %3264, %3265 : tensor<1x11008xf32>
    %3267 = stablehlo.transpose %arg4, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3268 = stablehlo.dot %3262, %3267, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x11008xf32>) -> tensor<1x11008xf32>
    %3269 = stablehlo.multiply %3266, %3268 : tensor<1x11008xf32>
    %3270 = stablehlo.transpose %arg3, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %3271 = stablehlo.dot %3269, %3270, precision = [DEFAULT, DEFAULT] : (tensor<1x11008xf32>, tensor<11008x4096xf32>) -> tensor<1x4096xf32>
    %3272 = stablehlo.add %3250, %3271 : tensor<1x4096xf32>
    %3273 = stablehlo.power %3272, %1 : tensor<1x4096xf32>
    %3274 = stablehlo.reduce(%3273 init: %4) across dimensions = [1] : (tensor<1x4096xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg362: tensor<f32>, %arg363: tensor<f32>)  {
      %3287 = stablehlo.add %arg362, %arg363 : tensor<f32>
      stablehlo.return %3287 : tensor<f32>
    }
    %3275 = stablehlo.multiply %3274, %0 : tensor<1xf32>
    %3276 = stablehlo.reshape %3275 : (tensor<1xf32>) -> tensor<1x1xf32>
    %3277 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %3278 = stablehlo.add %3276, %3277 : tensor<1x1xf32>
    %3279 = stablehlo.rsqrt %3278 : tensor<1x1xf32>
    %3280 = stablehlo.reshape %3279 : (tensor<1x1xf32>) -> tensor<1xf32>
    %3281 = stablehlo.broadcast_in_dim %3280, dims = [0] : (tensor<1xf32>) -> tensor<1x4096xf32>
    %3282 = stablehlo.multiply %3272, %3281 : tensor<1x4096xf32>
    %3283 = stablehlo.reshape %arg1 : (tensor<4096xf32>) -> tensor<1x4096xf32>
    %3284 = stablehlo.multiply %3282, %3283 : tensor<1x4096xf32>
    %3285 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,32000]{0,1}"} : (tensor<32000x4096xf32>) -> tensor<4096x32000xf32>
    %3286 = stablehlo.dot %3284, %3285, precision = [DEFAULT, DEFAULT] : (tensor<1x4096xf32>, tensor<4096x32000xf32>) -> tensor<1x32000xf32>
    return %3286, %58, %80, %160, %182, %262, %284, %364, %386, %466, %488, %568, %590, %670, %692, %772, %794, %874, %896, %976, %998, %1078, %1100, %1180, %1202, %1282, %1304, %1384, %1406, %1486, %1508, %1588, %1610, %1690, %1712, %1792, %1814, %1894, %1916, %1996, %2018, %2098, %2120, %2200, %2222, %2302, %2324, %2404, %2426, %2506, %2528, %2608, %2630, %2710, %2732, %2812, %2834, %2914, %2936, %3016, %3038, %3118, %3140, %3220, %3242 : tensor<1x32000xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>, tensor<2304x32x128xf32>
  }
}
