module @IrToHlo.7522 attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<32000x4096xf32>, %arg1: tensor<4096xf32>, %arg2: tensor<f32>, %arg3: tensor<4096x11008xf32>, %arg4: tensor<11008x4096xf32>, %arg5: tensor<4096xf32>, %arg6: tensor<4096x4096xf32>, %arg7: tensor<2048xi64>, %arg8: tensor<4096x4096xf32>, %arg9: tensor<4096xf32>, %arg10: tensor<4096x11008xf32>, %arg11: tensor<11008x4096xf32>, %arg12: tensor<4096xf32>, %arg13: tensor<4096x4096xf32>, %arg14: tensor<4096x4096xf32>, %arg15: tensor<4096xf32>, %arg16: tensor<4096x11008xf32>, %arg17: tensor<11008x4096xf32>, %arg18: tensor<4096xf32>, %arg19: tensor<4096x4096xf32>, %arg20: tensor<4096x4096xf32>, %arg21: tensor<4096xf32>, %arg22: tensor<4096x11008xf32>, %arg23: tensor<11008x4096xf32>, %arg24: tensor<4096xf32>, %arg25: tensor<4096x4096xf32>, %arg26: tensor<4096x4096xf32>, %arg27: tensor<4096xf32>, %arg28: tensor<4096x11008xf32>, %arg29: tensor<11008x4096xf32>, %arg30: tensor<4096xf32>, %arg31: tensor<4096x4096xf32>, %arg32: tensor<4096x4096xf32>, %arg33: tensor<4096xf32>, %arg34: tensor<4096x11008xf32>, %arg35: tensor<11008x4096xf32>, %arg36: tensor<4096xf32>, %arg37: tensor<4096x4096xf32>, %arg38: tensor<4096x4096xf32>, %arg39: tensor<4096xf32>, %arg40: tensor<4096x11008xf32>, %arg41: tensor<11008x4096xf32>, %arg42: tensor<4096xf32>, %arg43: tensor<4096x4096xf32>, %arg44: tensor<4096x4096xf32>, %arg45: tensor<4096xf32>, %arg46: tensor<4096x11008xf32>, %arg47: tensor<11008x4096xf32>, %arg48: tensor<4096xf32>, %arg49: tensor<4096x4096xf32>, %arg50: tensor<4096x4096xf32>, %arg51: tensor<4096xf32>, %arg52: tensor<4096x11008xf32>, %arg53: tensor<11008x4096xf32>, %arg54: tensor<4096xf32>, %arg55: tensor<4096x4096xf32>, %arg56: tensor<4096x4096xf32>, %arg57: tensor<4096xf32>, %arg58: tensor<4096x11008xf32>, %arg59: tensor<11008x4096xf32>, %arg60: tensor<4096xf32>, %arg61: tensor<4096x4096xf32>, %arg62: tensor<4096x4096xf32>, %arg63: tensor<4096xf32>, %arg64: tensor<4096x11008xf32>, %arg65: tensor<11008x4096xf32>, %arg66: tensor<4096xf32>, %arg67: tensor<4096x4096xf32>, %arg68: tensor<4096x4096xf32>, %arg69: tensor<4096xf32>, %arg70: tensor<4096x11008xf32>, %arg71: tensor<11008x4096xf32>, %arg72: tensor<4096xf32>, %arg73: tensor<4096x4096xf32>, %arg74: tensor<4096x4096xf32>, %arg75: tensor<4096xf32>, %arg76: tensor<4096x11008xf32>, %arg77: tensor<11008x4096xf32>, %arg78: tensor<4096xf32>, %arg79: tensor<4096x4096xf32>, %arg80: tensor<4096x4096xf32>, %arg81: tensor<4096xf32>, %arg82: tensor<4096x11008xf32>, %arg83: tensor<11008x4096xf32>, %arg84: tensor<4096xf32>, %arg85: tensor<4096x4096xf32>, %arg86: tensor<4096x4096xf32>, %arg87: tensor<4096xf32>, %arg88: tensor<4096x11008xf32>, %arg89: tensor<11008x4096xf32>, %arg90: tensor<4096xf32>, %arg91: tensor<4096x4096xf32>, %arg92: tensor<4096x4096xf32>, %arg93: tensor<4096xf32>, %arg94: tensor<4096x11008xf32>, %arg95: tensor<11008x4096xf32>, %arg96: tensor<4096xf32>, %arg97: tensor<4096x4096xf32>, %arg98: tensor<4096x4096xf32>, %arg99: tensor<4096xf32>, %arg100: tensor<4096x11008xf32>, %arg101: tensor<11008x4096xf32>, %arg102: tensor<4096xf32>, %arg103: tensor<4096x4096xf32>, %arg104: tensor<4096x4096xf32>, %arg105: tensor<4096xf32>, %arg106: tensor<4096x11008xf32>, %arg107: tensor<11008x4096xf32>, %arg108: tensor<4096xf32>, %arg109: tensor<4096x4096xf32>, %arg110: tensor<4096x4096xf32>, %arg111: tensor<4096xf32>, %arg112: tensor<4096x11008xf32>, %arg113: tensor<11008x4096xf32>, %arg114: tensor<4096xf32>, %arg115: tensor<4096x4096xf32>, %arg116: tensor<4096x4096xf32>, %arg117: tensor<4096xf32>, %arg118: tensor<4096x11008xf32>, %arg119: tensor<11008x4096xf32>, %arg120: tensor<4096xf32>, %arg121: tensor<4096x4096xf32>, %arg122: tensor<4096x4096xf32>, %arg123: tensor<4096xf32>, %arg124: tensor<4096x11008xf32>, %arg125: tensor<11008x4096xf32>, %arg126: tensor<4096xf32>, %arg127: tensor<4096x4096xf32>, %arg128: tensor<4096x4096xf32>, %arg129: tensor<4096xf32>, %arg130: tensor<4096x11008xf32>, %arg131: tensor<11008x4096xf32>, %arg132: tensor<4096xf32>, %arg133: tensor<4096x4096xf32>, %arg134: tensor<4096x4096xf32>, %arg135: tensor<4096xf32>, %arg136: tensor<4096x11008xf32>, %arg137: tensor<11008x4096xf32>, %arg138: tensor<4096xf32>, %arg139: tensor<4096x4096xf32>, %arg140: tensor<4096x4096xf32>, %arg141: tensor<4096xf32>, %arg142: tensor<4096x11008xf32>, %arg143: tensor<11008x4096xf32>, %arg144: tensor<4096xf32>, %arg145: tensor<4096x4096xf32>, %arg146: tensor<4096x4096xf32>, %arg147: tensor<4096xf32>, %arg148: tensor<4096x11008xf32>, %arg149: tensor<11008x4096xf32>, %arg150: tensor<4096xf32>, %arg151: tensor<4096x4096xf32>, %arg152: tensor<4096x4096xf32>, %arg153: tensor<4096xf32>, %arg154: tensor<4096x11008xf32>, %arg155: tensor<11008x4096xf32>, %arg156: tensor<4096xf32>, %arg157: tensor<4096x4096xf32>, %arg158: tensor<4096x4096xf32>, %arg159: tensor<4096xf32>, %arg160: tensor<4096x11008xf32>, %arg161: tensor<11008x4096xf32>, %arg162: tensor<4096xf32>, %arg163: tensor<4096x4096xf32>, %arg164: tensor<4096x4096xf32>, %arg165: tensor<4096xf32>, %arg166: tensor<4096x11008xf32>, %arg167: tensor<11008x4096xf32>, %arg168: tensor<4096xf32>, %arg169: tensor<4096x4096xf32>, %arg170: tensor<4096x4096xf32>, %arg171: tensor<4096xf32>, %arg172: tensor<4096x11008xf32>, %arg173: tensor<11008x4096xf32>, %arg174: tensor<4096xf32>, %arg175: tensor<4096x4096xf32>, %arg176: tensor<4096x4096xf32>, %arg177: tensor<4096xf32>, %arg178: tensor<4096x11008xf32>, %arg179: tensor<11008x4096xf32>, %arg180: tensor<4096xf32>, %arg181: tensor<4096x4096xf32>, %arg182: tensor<4096x4096xf32>, %arg183: tensor<4096xf32>, %arg184: tensor<4096x11008xf32>, %arg185: tensor<11008x4096xf32>, %arg186: tensor<4096xf32>, %arg187: tensor<4096x4096xf32>, %arg188: tensor<4096x4096xf32>, %arg189: tensor<4096xf32>, %arg190: tensor<4096x11008xf32>, %arg191: tensor<11008x4096xf32>, %arg192: tensor<4096xf32>, %arg193: tensor<4096x4096xf32>, %arg194: tensor<4096x4096xf32>, %arg195: tensor<4096xf32>, %arg196: tensor<2x1xi64>, %arg197: tensor<i64>, %arg198: tensor<32000x4096xf32>, %arg199: tensor<1xi64>, %arg200: tensor<i64>, %arg201: tensor<2x2304x32x128xf32>, %arg202: tensor<f32>, %arg203: tensor<4608x64xcomplex<f32>>, %arg204: tensor<4096x4096xf32>, %arg205: tensor<2x2304x32x128xf32>, %arg206: tensor<4096x4096xf32>, %arg207: tensor<11008x4096xf32>, %arg208: tensor<2x2304x32x128xf32>, %arg209: tensor<4096x4096xf32>, %arg210: tensor<2x2304x32x128xf32>, %arg211: tensor<4096x4096xf32>, %arg212: tensor<11008x4096xf32>, %arg213: tensor<2x2304x32x128xf32>, %arg214: tensor<4096x4096xf32>, %arg215: tensor<2x2304x32x128xf32>, %arg216: tensor<4096x4096xf32>, %arg217: tensor<11008x4096xf32>, %arg218: tensor<2x2304x32x128xf32>, %arg219: tensor<4096x4096xf32>, %arg220: tensor<2x2304x32x128xf32>, %arg221: tensor<4096x4096xf32>, %arg222: tensor<11008x4096xf32>, %arg223: tensor<2x2304x32x128xf32>, %arg224: tensor<4096x4096xf32>, %arg225: tensor<2x2304x32x128xf32>, %arg226: tensor<4096x4096xf32>, %arg227: tensor<11008x4096xf32>, %arg228: tensor<2x2304x32x128xf32>, %arg229: tensor<4096x4096xf32>, %arg230: tensor<2x2304x32x128xf32>, %arg231: tensor<4096x4096xf32>, %arg232: tensor<11008x4096xf32>, %arg233: tensor<2x2304x32x128xf32>, %arg234: tensor<4096x4096xf32>, %arg235: tensor<2x2304x32x128xf32>, %arg236: tensor<4096x4096xf32>, %arg237: tensor<11008x4096xf32>, %arg238: tensor<2x2304x32x128xf32>, %arg239: tensor<4096x4096xf32>, %arg240: tensor<2x2304x32x128xf32>, %arg241: tensor<4096x4096xf32>, %arg242: tensor<11008x4096xf32>, %arg243: tensor<2x2304x32x128xf32>, %arg244: tensor<4096x4096xf32>, %arg245: tensor<2x2304x32x128xf32>, %arg246: tensor<4096x4096xf32>, %arg247: tensor<11008x4096xf32>, %arg248: tensor<2x2304x32x128xf32>, %arg249: tensor<4096x4096xf32>, %arg250: tensor<2x2304x32x128xf32>, %arg251: tensor<4096x4096xf32>, %arg252: tensor<11008x4096xf32>, %arg253: tensor<2x2304x32x128xf32>, %arg254: tensor<4096x4096xf32>, %arg255: tensor<2x2304x32x128xf32>, %arg256: tensor<4096x4096xf32>, %arg257: tensor<11008x4096xf32>, %arg258: tensor<2x2304x32x128xf32>, %arg259: tensor<4096x4096xf32>, %arg260: tensor<2x2304x32x128xf32>, %arg261: tensor<4096x4096xf32>, %arg262: tensor<11008x4096xf32>, %arg263: tensor<2x2304x32x128xf32>, %arg264: tensor<4096x4096xf32>, %arg265: tensor<2x2304x32x128xf32>, %arg266: tensor<4096x4096xf32>, %arg267: tensor<11008x4096xf32>, %arg268: tensor<2x2304x32x128xf32>, %arg269: tensor<4096x4096xf32>, %arg270: tensor<2x2304x32x128xf32>, %arg271: tensor<4096x4096xf32>, %arg272: tensor<11008x4096xf32>, %arg273: tensor<2x2304x32x128xf32>, %arg274: tensor<4096x4096xf32>, %arg275: tensor<2x2304x32x128xf32>, %arg276: tensor<4096x4096xf32>, %arg277: tensor<11008x4096xf32>, %arg278: tensor<2x2304x32x128xf32>, %arg279: tensor<4096x4096xf32>, %arg280: tensor<2x2304x32x128xf32>, %arg281: tensor<4096x4096xf32>, %arg282: tensor<11008x4096xf32>, %arg283: tensor<2x2304x32x128xf32>, %arg284: tensor<4096x4096xf32>, %arg285: tensor<2x2304x32x128xf32>, %arg286: tensor<4096x4096xf32>, %arg287: tensor<11008x4096xf32>, %arg288: tensor<2x2304x32x128xf32>, %arg289: tensor<4096x4096xf32>, %arg290: tensor<2x2304x32x128xf32>, %arg291: tensor<4096x4096xf32>, %arg292: tensor<11008x4096xf32>, %arg293: tensor<2x2304x32x128xf32>, %arg294: tensor<4096x4096xf32>, %arg295: tensor<2x2304x32x128xf32>, %arg296: tensor<4096x4096xf32>, %arg297: tensor<11008x4096xf32>, %arg298: tensor<2x2304x32x128xf32>, %arg299: tensor<4096x4096xf32>, %arg300: tensor<2x2304x32x128xf32>, %arg301: tensor<4096x4096xf32>, %arg302: tensor<11008x4096xf32>, %arg303: tensor<2x2304x32x128xf32>, %arg304: tensor<4096x4096xf32>, %arg305: tensor<2x2304x32x128xf32>, %arg306: tensor<4096x4096xf32>, %arg307: tensor<11008x4096xf32>, %arg308: tensor<2x2304x32x128xf32>, %arg309: tensor<4096x4096xf32>, %arg310: tensor<2x2304x32x128xf32>, %arg311: tensor<4096x4096xf32>, %arg312: tensor<11008x4096xf32>, %arg313: tensor<2x2304x32x128xf32>, %arg314: tensor<4096x4096xf32>, %arg315: tensor<2x2304x32x128xf32>, %arg316: tensor<4096x4096xf32>, %arg317: tensor<11008x4096xf32>, %arg318: tensor<2x2304x32x128xf32>, %arg319: tensor<4096x4096xf32>, %arg320: tensor<2x2304x32x128xf32>, %arg321: tensor<4096x4096xf32>, %arg322: tensor<11008x4096xf32>, %arg323: tensor<2x2304x32x128xf32>, %arg324: tensor<4096x4096xf32>, %arg325: tensor<2x2304x32x128xf32>, %arg326: tensor<4096x4096xf32>, %arg327: tensor<11008x4096xf32>, %arg328: tensor<2x2304x32x128xf32>, %arg329: tensor<4096x4096xf32>, %arg330: tensor<2x2304x32x128xf32>, %arg331: tensor<4096x4096xf32>, %arg332: tensor<11008x4096xf32>, %arg333: tensor<2x2304x32x128xf32>, %arg334: tensor<4096x4096xf32>, %arg335: tensor<2x2304x32x128xf32>, %arg336: tensor<4096x4096xf32>, %arg337: tensor<11008x4096xf32>, %arg338: tensor<2x2304x32x128xf32>, %arg339: tensor<4096x4096xf32>, %arg340: tensor<2x2304x32x128xf32>, %arg341: tensor<4096x4096xf32>, %arg342: tensor<11008x4096xf32>, %arg343: tensor<2x2304x32x128xf32>, %arg344: tensor<4096x4096xf32>, %arg345: tensor<2x2304x32x128xf32>, %arg346: tensor<4096x4096xf32>, %arg347: tensor<11008x4096xf32>, %arg348: tensor<2x2304x32x128xf32>, %arg349: tensor<4096x4096xf32>, %arg350: tensor<2x2304x32x128xf32>, %arg351: tensor<4096x4096xf32>, %arg352: tensor<11008x4096xf32>, %arg353: tensor<2x2304x32x128xf32>, %arg354: tensor<4096x4096xf32>, %arg355: tensor<2x2304x32x128xf32>, %arg356: tensor<4096x4096xf32>, %arg357: tensor<11008x4096xf32>, %arg358: tensor<2x2304x32x128xf32>, %arg359: tensor<4096x4096xf32>, %arg360: tensor<2x2304x32x128xf32>, %arg361: tensor<4096x4096xf32>, %arg362: tensor<11008x4096xf32>) -> (tensor<2x1x32000xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>) {
    %0 = stablehlo.constant dense<2.44140625E-4> : tensor<2x1xf32>
    %1 = stablehlo.constant dense<2.000000e+00> : tensor<2x1x4096xf32>
    %2 = stablehlo.constant dense<0> : tensor<1xi64>
    %3 = stablehlo.constant dense<0> : tensor<2x1xi64>
    %4 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.compare  LT, %arg196, %3 : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x1xi1>
    %7 = stablehlo.reshape %arg197 : (tensor<i64>) -> tensor<1xi64>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1] : (tensor<1xi64>) -> tensor<2x1xi64>
    %9 = stablehlo.add %arg196, %8 : tensor<2x1xi64>
    %10 = stablehlo.select %6, %9, %arg196 : tensor<2x1xi1>, tensor<2x1xi64>
    %11 = stablehlo.reshape %10 : (tensor<2x1xi64>) -> tensor<2x1x1xi64>
    %12 = "stablehlo.gather"(%arg198, %11) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 4096]> : tensor<2xi64>} : (tensor<32000x4096xf32>, tensor<2x1x1xi64>) -> tensor<2x1x4096xf32>
    %13 = stablehlo.power %12, %1 : tensor<2x1x4096xf32>
    %14 = stablehlo.reduce(%13 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %15 = stablehlo.multiply %14, %0 : tensor<2x1xf32>
    %16 = stablehlo.reshape %15 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %17 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %18 = stablehlo.add %16, %17 : tensor<2x1x1xf32>
    %19 = stablehlo.rsqrt %18 : tensor<2x1x1xf32>
    %20 = stablehlo.reshape %19 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %22 = stablehlo.multiply %12, %21 : tensor<2x1x4096xf32>
    %23 = stablehlo.broadcast_in_dim %arg195, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<2x1x4096xf32>
    %25 = stablehlo.reshape %24 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %26 = stablehlo.transpose %arg206, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %27 = stablehlo.dot %25, %26, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %28 = stablehlo.reshape %27 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %29 = stablehlo.slice %28 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %30 = stablehlo.reshape %29 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %31 = stablehlo.slice %28 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %32 = stablehlo.reshape %31 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %33 = stablehlo.complex %30, %32 : tensor<2x1x32x64xcomplex<f32>>
    %34 = stablehlo.convert %arg199 : (tensor<1xi64>) -> tensor<1xui32>
    %35 = "stablehlo.gather"(%arg203, %34) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 64]> : tensor<2xi64>} : (tensor<4608x64xcomplex<f32>>, tensor<1xui32>) -> tensor<1x64xcomplex<f32>>
    %36 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %37 = stablehlo.multiply %33, %36 : tensor<2x1x32x64xcomplex<f32>>
    %38 = stablehlo.real %37 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %39 = stablehlo.reshape %38 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %40 = stablehlo.imag %37 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %41 = stablehlo.reshape %40 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %42 = stablehlo.concatenate %39, %41, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %43 = stablehlo.reshape %42 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %44 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %45 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %46 = stablehlo.add %arg199, %45 : tensor<1xi64>
    %47 = stablehlo.select %44, %46, %arg199 : tensor<1xi1>, tensor<1xi64>
    %48 = stablehlo.reshape %47 : (tensor<1xi64>) -> tensor<1x1xi64>
    %49 = stablehlo.reshape %24 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %50 = stablehlo.transpose %arg204, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %51 = stablehlo.dot %49, %50, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %52 = stablehlo.reshape %51 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %53 = stablehlo.slice %52 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %54 = stablehlo.reshape %53 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %55 = stablehlo.slice %52 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %56 = stablehlo.reshape %55 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %57 = stablehlo.complex %54, %56 : tensor<2x1x32x64xcomplex<f32>>
    %58 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %59 = stablehlo.multiply %57, %58 : tensor<2x1x32x64xcomplex<f32>>
    %60 = stablehlo.real %59 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %61 = stablehlo.reshape %60 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %62 = stablehlo.imag %59 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %63 = stablehlo.reshape %62 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %64 = stablehlo.concatenate %61, %63, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %65 = stablehlo.reshape %64 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %66 = "stablehlo.scatter"(%arg205, %48, %65) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %67 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %68 = "stablehlo.gather"(%66, %67) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %69 = stablehlo.transpose %68, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %70 = stablehlo.reshape %69 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %71 = stablehlo.dot_general %43, %70, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %72 = stablehlo.reshape %71 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %73 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %74 = stablehlo.divide %72, %73 : tensor<2x32x1x2048xf32>
    %75 = stablehlo.reduce(%74 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %77 = stablehlo.subtract %74, %76 : tensor<2x32x1x2048xf32>
    %78 = stablehlo.exponential %77 : tensor<2x32x1x2048xf32>
    %79 = stablehlo.reduce(%78 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %80 = stablehlo.broadcast_in_dim %79, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %81 = stablehlo.divide %78, %80 : tensor<2x32x1x2048xf32>
    %82 = stablehlo.reshape %81 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %83 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %84 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %85 = stablehlo.add %arg199, %84 : tensor<1xi64>
    %86 = stablehlo.select %83, %85, %arg199 : tensor<1xi1>, tensor<1xi64>
    %87 = stablehlo.reshape %86 : (tensor<1xi64>) -> tensor<1x1xi64>
    %88 = stablehlo.reshape %24 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %89 = stablehlo.transpose %arg194, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %90 = stablehlo.dot %88, %89, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %91 = stablehlo.reshape %90 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %92 = "stablehlo.scatter"(%arg201, %87, %91) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %93 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %94 = "stablehlo.gather"(%92, %93) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %95 = stablehlo.transpose %94, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %96 = stablehlo.reshape %95 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %97 = stablehlo.dot_general %82, %96, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %98 = stablehlo.reshape %97 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %99 = stablehlo.transpose %arg193, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %100 = stablehlo.dot %98, %99, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %101 = stablehlo.reshape %100 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %102 = stablehlo.add %12, %101 : tensor<2x1x4096xf32>
    %103 = stablehlo.power %102, %1 : tensor<2x1x4096xf32>
    %104 = stablehlo.reduce(%103 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %105 = stablehlo.multiply %104, %0 : tensor<2x1xf32>
    %106 = stablehlo.reshape %105 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %107 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %108 = stablehlo.add %106, %107 : tensor<2x1x1xf32>
    %109 = stablehlo.rsqrt %108 : tensor<2x1x1xf32>
    %110 = stablehlo.reshape %109 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %111 = stablehlo.broadcast_in_dim %110, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %112 = stablehlo.multiply %102, %111 : tensor<2x1x4096xf32>
    %113 = stablehlo.broadcast_in_dim %arg192, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %114 = stablehlo.multiply %112, %113 : tensor<2x1x4096xf32>
    %115 = stablehlo.reshape %114 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %116 = stablehlo.transpose %arg207, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %117 = stablehlo.dot %115, %116, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %118 = stablehlo.reshape %117 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %119 = stablehlo.logistic %118 : tensor<2x1x11008xf32>
    %120 = stablehlo.multiply %118, %119 : tensor<2x1x11008xf32>
    %121 = stablehlo.reshape %114 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %122 = stablehlo.transpose %arg191, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %123 = stablehlo.dot %121, %122, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %124 = stablehlo.reshape %123 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %125 = stablehlo.multiply %120, %124 : tensor<2x1x11008xf32>
    %126 = stablehlo.reshape %125 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %127 = stablehlo.transpose %arg190, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %128 = stablehlo.dot %126, %127, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %129 = stablehlo.reshape %128 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %130 = stablehlo.add %102, %129 : tensor<2x1x4096xf32>
    %131 = stablehlo.power %130, %1 : tensor<2x1x4096xf32>
    %132 = stablehlo.reduce(%131 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %133 = stablehlo.multiply %132, %0 : tensor<2x1xf32>
    %134 = stablehlo.reshape %133 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %135 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %136 = stablehlo.add %134, %135 : tensor<2x1x1xf32>
    %137 = stablehlo.rsqrt %136 : tensor<2x1x1xf32>
    %138 = stablehlo.reshape %137 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %139 = stablehlo.broadcast_in_dim %138, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %140 = stablehlo.multiply %130, %139 : tensor<2x1x4096xf32>
    %141 = stablehlo.broadcast_in_dim %arg189, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %142 = stablehlo.multiply %140, %141 : tensor<2x1x4096xf32>
    %143 = stablehlo.reshape %142 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %144 = stablehlo.transpose %arg211, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %145 = stablehlo.dot %143, %144, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %146 = stablehlo.reshape %145 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %147 = stablehlo.slice %146 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %148 = stablehlo.reshape %147 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %149 = stablehlo.slice %146 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %150 = stablehlo.reshape %149 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %151 = stablehlo.complex %148, %150 : tensor<2x1x32x64xcomplex<f32>>
    %152 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %153 = stablehlo.multiply %151, %152 : tensor<2x1x32x64xcomplex<f32>>
    %154 = stablehlo.real %153 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %155 = stablehlo.reshape %154 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %156 = stablehlo.imag %153 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %157 = stablehlo.reshape %156 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %158 = stablehlo.concatenate %155, %157, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %159 = stablehlo.reshape %158 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %160 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %161 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %162 = stablehlo.add %arg199, %161 : tensor<1xi64>
    %163 = stablehlo.select %160, %162, %arg199 : tensor<1xi1>, tensor<1xi64>
    %164 = stablehlo.reshape %163 : (tensor<1xi64>) -> tensor<1x1xi64>
    %165 = stablehlo.reshape %142 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %166 = stablehlo.transpose %arg209, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %167 = stablehlo.dot %165, %166, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %168 = stablehlo.reshape %167 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %169 = stablehlo.slice %168 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %170 = stablehlo.reshape %169 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %171 = stablehlo.slice %168 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %172 = stablehlo.reshape %171 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %173 = stablehlo.complex %170, %172 : tensor<2x1x32x64xcomplex<f32>>
    %174 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %175 = stablehlo.multiply %173, %174 : tensor<2x1x32x64xcomplex<f32>>
    %176 = stablehlo.real %175 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %177 = stablehlo.reshape %176 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %178 = stablehlo.imag %175 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %179 = stablehlo.reshape %178 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %180 = stablehlo.concatenate %177, %179, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %181 = stablehlo.reshape %180 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %182 = "stablehlo.scatter"(%arg210, %164, %181) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %183 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %184 = "stablehlo.gather"(%182, %183) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %185 = stablehlo.transpose %184, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %186 = stablehlo.reshape %185 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %187 = stablehlo.dot_general %159, %186, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %188 = stablehlo.reshape %187 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %189 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %190 = stablehlo.divide %188, %189 : tensor<2x32x1x2048xf32>
    %191 = stablehlo.reduce(%190 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %192 = stablehlo.broadcast_in_dim %191, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %193 = stablehlo.subtract %190, %192 : tensor<2x32x1x2048xf32>
    %194 = stablehlo.exponential %193 : tensor<2x32x1x2048xf32>
    %195 = stablehlo.reduce(%194 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %196 = stablehlo.broadcast_in_dim %195, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %197 = stablehlo.divide %194, %196 : tensor<2x32x1x2048xf32>
    %198 = stablehlo.reshape %197 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %199 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %200 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %201 = stablehlo.add %arg199, %200 : tensor<1xi64>
    %202 = stablehlo.select %199, %201, %arg199 : tensor<1xi1>, tensor<1xi64>
    %203 = stablehlo.reshape %202 : (tensor<1xi64>) -> tensor<1x1xi64>
    %204 = stablehlo.reshape %142 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %205 = stablehlo.transpose %arg188, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %206 = stablehlo.dot %204, %205, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %207 = stablehlo.reshape %206 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %208 = "stablehlo.scatter"(%arg208, %203, %207) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %209 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %210 = "stablehlo.gather"(%208, %209) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %211 = stablehlo.transpose %210, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %212 = stablehlo.reshape %211 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %213 = stablehlo.dot_general %198, %212, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %214 = stablehlo.reshape %213 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %215 = stablehlo.transpose %arg187, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %216 = stablehlo.dot %214, %215, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %217 = stablehlo.reshape %216 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %218 = stablehlo.add %130, %217 : tensor<2x1x4096xf32>
    %219 = stablehlo.power %218, %1 : tensor<2x1x4096xf32>
    %220 = stablehlo.reduce(%219 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %221 = stablehlo.multiply %220, %0 : tensor<2x1xf32>
    %222 = stablehlo.reshape %221 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %223 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %224 = stablehlo.add %222, %223 : tensor<2x1x1xf32>
    %225 = stablehlo.rsqrt %224 : tensor<2x1x1xf32>
    %226 = stablehlo.reshape %225 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %227 = stablehlo.broadcast_in_dim %226, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %228 = stablehlo.multiply %218, %227 : tensor<2x1x4096xf32>
    %229 = stablehlo.broadcast_in_dim %arg186, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %230 = stablehlo.multiply %228, %229 : tensor<2x1x4096xf32>
    %231 = stablehlo.reshape %230 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %232 = stablehlo.transpose %arg212, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %233 = stablehlo.dot %231, %232, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %234 = stablehlo.reshape %233 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %235 = stablehlo.logistic %234 : tensor<2x1x11008xf32>
    %236 = stablehlo.multiply %234, %235 : tensor<2x1x11008xf32>
    %237 = stablehlo.reshape %230 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %238 = stablehlo.transpose %arg185, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %239 = stablehlo.dot %237, %238, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %240 = stablehlo.reshape %239 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %241 = stablehlo.multiply %236, %240 : tensor<2x1x11008xf32>
    %242 = stablehlo.reshape %241 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %243 = stablehlo.transpose %arg184, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %244 = stablehlo.dot %242, %243, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %245 = stablehlo.reshape %244 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %246 = stablehlo.add %218, %245 : tensor<2x1x4096xf32>
    %247 = stablehlo.power %246, %1 : tensor<2x1x4096xf32>
    %248 = stablehlo.reduce(%247 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %249 = stablehlo.multiply %248, %0 : tensor<2x1xf32>
    %250 = stablehlo.reshape %249 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %251 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %252 = stablehlo.add %250, %251 : tensor<2x1x1xf32>
    %253 = stablehlo.rsqrt %252 : tensor<2x1x1xf32>
    %254 = stablehlo.reshape %253 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %255 = stablehlo.broadcast_in_dim %254, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %256 = stablehlo.multiply %246, %255 : tensor<2x1x4096xf32>
    %257 = stablehlo.broadcast_in_dim %arg183, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %258 = stablehlo.multiply %256, %257 : tensor<2x1x4096xf32>
    %259 = stablehlo.reshape %258 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %260 = stablehlo.transpose %arg216, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %261 = stablehlo.dot %259, %260, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %262 = stablehlo.reshape %261 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %263 = stablehlo.slice %262 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %264 = stablehlo.reshape %263 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %265 = stablehlo.slice %262 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %266 = stablehlo.reshape %265 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %267 = stablehlo.complex %264, %266 : tensor<2x1x32x64xcomplex<f32>>
    %268 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %269 = stablehlo.multiply %267, %268 : tensor<2x1x32x64xcomplex<f32>>
    %270 = stablehlo.real %269 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %271 = stablehlo.reshape %270 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %272 = stablehlo.imag %269 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %273 = stablehlo.reshape %272 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %274 = stablehlo.concatenate %271, %273, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %275 = stablehlo.reshape %274 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %276 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %277 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %278 = stablehlo.add %arg199, %277 : tensor<1xi64>
    %279 = stablehlo.select %276, %278, %arg199 : tensor<1xi1>, tensor<1xi64>
    %280 = stablehlo.reshape %279 : (tensor<1xi64>) -> tensor<1x1xi64>
    %281 = stablehlo.reshape %258 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %282 = stablehlo.transpose %arg214, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %283 = stablehlo.dot %281, %282, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %284 = stablehlo.reshape %283 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %285 = stablehlo.slice %284 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %286 = stablehlo.reshape %285 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %287 = stablehlo.slice %284 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %288 = stablehlo.reshape %287 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %289 = stablehlo.complex %286, %288 : tensor<2x1x32x64xcomplex<f32>>
    %290 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %291 = stablehlo.multiply %289, %290 : tensor<2x1x32x64xcomplex<f32>>
    %292 = stablehlo.real %291 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %293 = stablehlo.reshape %292 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %294 = stablehlo.imag %291 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %295 = stablehlo.reshape %294 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %296 = stablehlo.concatenate %293, %295, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %297 = stablehlo.reshape %296 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %298 = "stablehlo.scatter"(%arg215, %280, %297) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %299 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %300 = "stablehlo.gather"(%298, %299) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %301 = stablehlo.transpose %300, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %302 = stablehlo.reshape %301 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %303 = stablehlo.dot_general %275, %302, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %304 = stablehlo.reshape %303 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %305 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %306 = stablehlo.divide %304, %305 : tensor<2x32x1x2048xf32>
    %307 = stablehlo.reduce(%306 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %309 = stablehlo.subtract %306, %308 : tensor<2x32x1x2048xf32>
    %310 = stablehlo.exponential %309 : tensor<2x32x1x2048xf32>
    %311 = stablehlo.reduce(%310 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %313 = stablehlo.divide %310, %312 : tensor<2x32x1x2048xf32>
    %314 = stablehlo.reshape %313 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %315 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %316 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %317 = stablehlo.add %arg199, %316 : tensor<1xi64>
    %318 = stablehlo.select %315, %317, %arg199 : tensor<1xi1>, tensor<1xi64>
    %319 = stablehlo.reshape %318 : (tensor<1xi64>) -> tensor<1x1xi64>
    %320 = stablehlo.reshape %258 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %321 = stablehlo.transpose %arg182, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %322 = stablehlo.dot %320, %321, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %323 = stablehlo.reshape %322 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %324 = "stablehlo.scatter"(%arg213, %319, %323) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %325 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %326 = "stablehlo.gather"(%324, %325) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %327 = stablehlo.transpose %326, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %328 = stablehlo.reshape %327 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %329 = stablehlo.dot_general %314, %328, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %330 = stablehlo.reshape %329 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %331 = stablehlo.transpose %arg181, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %332 = stablehlo.dot %330, %331, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %333 = stablehlo.reshape %332 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %334 = stablehlo.add %246, %333 : tensor<2x1x4096xf32>
    %335 = stablehlo.power %334, %1 : tensor<2x1x4096xf32>
    %336 = stablehlo.reduce(%335 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %337 = stablehlo.multiply %336, %0 : tensor<2x1xf32>
    %338 = stablehlo.reshape %337 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %339 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %340 = stablehlo.add %338, %339 : tensor<2x1x1xf32>
    %341 = stablehlo.rsqrt %340 : tensor<2x1x1xf32>
    %342 = stablehlo.reshape %341 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %343 = stablehlo.broadcast_in_dim %342, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %344 = stablehlo.multiply %334, %343 : tensor<2x1x4096xf32>
    %345 = stablehlo.broadcast_in_dim %arg180, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %346 = stablehlo.multiply %344, %345 : tensor<2x1x4096xf32>
    %347 = stablehlo.reshape %346 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %348 = stablehlo.transpose %arg217, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %349 = stablehlo.dot %347, %348, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %350 = stablehlo.reshape %349 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %351 = stablehlo.logistic %350 : tensor<2x1x11008xf32>
    %352 = stablehlo.multiply %350, %351 : tensor<2x1x11008xf32>
    %353 = stablehlo.reshape %346 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %354 = stablehlo.transpose %arg179, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %355 = stablehlo.dot %353, %354, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %356 = stablehlo.reshape %355 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %357 = stablehlo.multiply %352, %356 : tensor<2x1x11008xf32>
    %358 = stablehlo.reshape %357 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %359 = stablehlo.transpose %arg178, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %360 = stablehlo.dot %358, %359, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %361 = stablehlo.reshape %360 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %362 = stablehlo.add %334, %361 : tensor<2x1x4096xf32>
    %363 = stablehlo.power %362, %1 : tensor<2x1x4096xf32>
    %364 = stablehlo.reduce(%363 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %365 = stablehlo.multiply %364, %0 : tensor<2x1xf32>
    %366 = stablehlo.reshape %365 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %367 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %368 = stablehlo.add %366, %367 : tensor<2x1x1xf32>
    %369 = stablehlo.rsqrt %368 : tensor<2x1x1xf32>
    %370 = stablehlo.reshape %369 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %371 = stablehlo.broadcast_in_dim %370, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %372 = stablehlo.multiply %362, %371 : tensor<2x1x4096xf32>
    %373 = stablehlo.broadcast_in_dim %arg177, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %374 = stablehlo.multiply %372, %373 : tensor<2x1x4096xf32>
    %375 = stablehlo.reshape %374 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %376 = stablehlo.transpose %arg221, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %377 = stablehlo.dot %375, %376, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %378 = stablehlo.reshape %377 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %379 = stablehlo.slice %378 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %380 = stablehlo.reshape %379 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %381 = stablehlo.slice %378 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %382 = stablehlo.reshape %381 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %383 = stablehlo.complex %380, %382 : tensor<2x1x32x64xcomplex<f32>>
    %384 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %385 = stablehlo.multiply %383, %384 : tensor<2x1x32x64xcomplex<f32>>
    %386 = stablehlo.real %385 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %387 = stablehlo.reshape %386 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %388 = stablehlo.imag %385 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %389 = stablehlo.reshape %388 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %390 = stablehlo.concatenate %387, %389, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %391 = stablehlo.reshape %390 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %392 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %393 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %394 = stablehlo.add %arg199, %393 : tensor<1xi64>
    %395 = stablehlo.select %392, %394, %arg199 : tensor<1xi1>, tensor<1xi64>
    %396 = stablehlo.reshape %395 : (tensor<1xi64>) -> tensor<1x1xi64>
    %397 = stablehlo.reshape %374 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %398 = stablehlo.transpose %arg219, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %399 = stablehlo.dot %397, %398, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %400 = stablehlo.reshape %399 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %401 = stablehlo.slice %400 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %402 = stablehlo.reshape %401 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %403 = stablehlo.slice %400 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %404 = stablehlo.reshape %403 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %405 = stablehlo.complex %402, %404 : tensor<2x1x32x64xcomplex<f32>>
    %406 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %407 = stablehlo.multiply %405, %406 : tensor<2x1x32x64xcomplex<f32>>
    %408 = stablehlo.real %407 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %409 = stablehlo.reshape %408 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %410 = stablehlo.imag %407 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %411 = stablehlo.reshape %410 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %412 = stablehlo.concatenate %409, %411, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %413 = stablehlo.reshape %412 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %414 = "stablehlo.scatter"(%arg220, %396, %413) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %415 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %416 = "stablehlo.gather"(%414, %415) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %417 = stablehlo.transpose %416, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %418 = stablehlo.reshape %417 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %419 = stablehlo.dot_general %391, %418, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %420 = stablehlo.reshape %419 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %421 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %422 = stablehlo.divide %420, %421 : tensor<2x32x1x2048xf32>
    %423 = stablehlo.reduce(%422 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %424 = stablehlo.broadcast_in_dim %423, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %425 = stablehlo.subtract %422, %424 : tensor<2x32x1x2048xf32>
    %426 = stablehlo.exponential %425 : tensor<2x32x1x2048xf32>
    %427 = stablehlo.reduce(%426 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %428 = stablehlo.broadcast_in_dim %427, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %429 = stablehlo.divide %426, %428 : tensor<2x32x1x2048xf32>
    %430 = stablehlo.reshape %429 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %431 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %432 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %433 = stablehlo.add %arg199, %432 : tensor<1xi64>
    %434 = stablehlo.select %431, %433, %arg199 : tensor<1xi1>, tensor<1xi64>
    %435 = stablehlo.reshape %434 : (tensor<1xi64>) -> tensor<1x1xi64>
    %436 = stablehlo.reshape %374 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %437 = stablehlo.transpose %arg176, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %438 = stablehlo.dot %436, %437, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %439 = stablehlo.reshape %438 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %440 = "stablehlo.scatter"(%arg218, %435, %439) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %441 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %442 = "stablehlo.gather"(%440, %441) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %443 = stablehlo.transpose %442, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %444 = stablehlo.reshape %443 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %445 = stablehlo.dot_general %430, %444, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %446 = stablehlo.reshape %445 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %447 = stablehlo.transpose %arg175, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %448 = stablehlo.dot %446, %447, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %449 = stablehlo.reshape %448 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %450 = stablehlo.add %362, %449 : tensor<2x1x4096xf32>
    %451 = stablehlo.power %450, %1 : tensor<2x1x4096xf32>
    %452 = stablehlo.reduce(%451 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %453 = stablehlo.multiply %452, %0 : tensor<2x1xf32>
    %454 = stablehlo.reshape %453 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %455 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %456 = stablehlo.add %454, %455 : tensor<2x1x1xf32>
    %457 = stablehlo.rsqrt %456 : tensor<2x1x1xf32>
    %458 = stablehlo.reshape %457 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %459 = stablehlo.broadcast_in_dim %458, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %460 = stablehlo.multiply %450, %459 : tensor<2x1x4096xf32>
    %461 = stablehlo.broadcast_in_dim %arg174, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %462 = stablehlo.multiply %460, %461 : tensor<2x1x4096xf32>
    %463 = stablehlo.reshape %462 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %464 = stablehlo.transpose %arg222, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %465 = stablehlo.dot %463, %464, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %466 = stablehlo.reshape %465 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %467 = stablehlo.logistic %466 : tensor<2x1x11008xf32>
    %468 = stablehlo.multiply %466, %467 : tensor<2x1x11008xf32>
    %469 = stablehlo.reshape %462 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %470 = stablehlo.transpose %arg173, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %471 = stablehlo.dot %469, %470, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %472 = stablehlo.reshape %471 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %473 = stablehlo.multiply %468, %472 : tensor<2x1x11008xf32>
    %474 = stablehlo.reshape %473 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %475 = stablehlo.transpose %arg172, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %476 = stablehlo.dot %474, %475, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %477 = stablehlo.reshape %476 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %478 = stablehlo.add %450, %477 : tensor<2x1x4096xf32>
    %479 = stablehlo.power %478, %1 : tensor<2x1x4096xf32>
    %480 = stablehlo.reduce(%479 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %481 = stablehlo.multiply %480, %0 : tensor<2x1xf32>
    %482 = stablehlo.reshape %481 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %483 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %484 = stablehlo.add %482, %483 : tensor<2x1x1xf32>
    %485 = stablehlo.rsqrt %484 : tensor<2x1x1xf32>
    %486 = stablehlo.reshape %485 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %487 = stablehlo.broadcast_in_dim %486, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %488 = stablehlo.multiply %478, %487 : tensor<2x1x4096xf32>
    %489 = stablehlo.broadcast_in_dim %arg171, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %490 = stablehlo.multiply %488, %489 : tensor<2x1x4096xf32>
    %491 = stablehlo.reshape %490 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %492 = stablehlo.transpose %arg226, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %493 = stablehlo.dot %491, %492, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %494 = stablehlo.reshape %493 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %495 = stablehlo.slice %494 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %496 = stablehlo.reshape %495 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %497 = stablehlo.slice %494 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %498 = stablehlo.reshape %497 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %499 = stablehlo.complex %496, %498 : tensor<2x1x32x64xcomplex<f32>>
    %500 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %501 = stablehlo.multiply %499, %500 : tensor<2x1x32x64xcomplex<f32>>
    %502 = stablehlo.real %501 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %503 = stablehlo.reshape %502 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %504 = stablehlo.imag %501 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %505 = stablehlo.reshape %504 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %506 = stablehlo.concatenate %503, %505, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %507 = stablehlo.reshape %506 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %508 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %509 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %510 = stablehlo.add %arg199, %509 : tensor<1xi64>
    %511 = stablehlo.select %508, %510, %arg199 : tensor<1xi1>, tensor<1xi64>
    %512 = stablehlo.reshape %511 : (tensor<1xi64>) -> tensor<1x1xi64>
    %513 = stablehlo.reshape %490 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %514 = stablehlo.transpose %arg224, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %515 = stablehlo.dot %513, %514, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %516 = stablehlo.reshape %515 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %517 = stablehlo.slice %516 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %518 = stablehlo.reshape %517 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %519 = stablehlo.slice %516 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %520 = stablehlo.reshape %519 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %521 = stablehlo.complex %518, %520 : tensor<2x1x32x64xcomplex<f32>>
    %522 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %523 = stablehlo.multiply %521, %522 : tensor<2x1x32x64xcomplex<f32>>
    %524 = stablehlo.real %523 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %525 = stablehlo.reshape %524 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %526 = stablehlo.imag %523 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %527 = stablehlo.reshape %526 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %528 = stablehlo.concatenate %525, %527, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %529 = stablehlo.reshape %528 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %530 = "stablehlo.scatter"(%arg225, %512, %529) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %531 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %532 = "stablehlo.gather"(%530, %531) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %533 = stablehlo.transpose %532, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %534 = stablehlo.reshape %533 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %535 = stablehlo.dot_general %507, %534, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %536 = stablehlo.reshape %535 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %537 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %538 = stablehlo.divide %536, %537 : tensor<2x32x1x2048xf32>
    %539 = stablehlo.reduce(%538 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %540 = stablehlo.broadcast_in_dim %539, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %541 = stablehlo.subtract %538, %540 : tensor<2x32x1x2048xf32>
    %542 = stablehlo.exponential %541 : tensor<2x32x1x2048xf32>
    %543 = stablehlo.reduce(%542 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %544 = stablehlo.broadcast_in_dim %543, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %545 = stablehlo.divide %542, %544 : tensor<2x32x1x2048xf32>
    %546 = stablehlo.reshape %545 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %547 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %548 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %549 = stablehlo.add %arg199, %548 : tensor<1xi64>
    %550 = stablehlo.select %547, %549, %arg199 : tensor<1xi1>, tensor<1xi64>
    %551 = stablehlo.reshape %550 : (tensor<1xi64>) -> tensor<1x1xi64>
    %552 = stablehlo.reshape %490 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %553 = stablehlo.transpose %arg170, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %554 = stablehlo.dot %552, %553, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %555 = stablehlo.reshape %554 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %556 = "stablehlo.scatter"(%arg223, %551, %555) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %557 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %558 = "stablehlo.gather"(%556, %557) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %559 = stablehlo.transpose %558, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %560 = stablehlo.reshape %559 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %561 = stablehlo.dot_general %546, %560, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %562 = stablehlo.reshape %561 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %563 = stablehlo.transpose %arg169, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %564 = stablehlo.dot %562, %563, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %565 = stablehlo.reshape %564 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %566 = stablehlo.add %478, %565 : tensor<2x1x4096xf32>
    %567 = stablehlo.power %566, %1 : tensor<2x1x4096xf32>
    %568 = stablehlo.reduce(%567 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %569 = stablehlo.multiply %568, %0 : tensor<2x1xf32>
    %570 = stablehlo.reshape %569 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %571 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %572 = stablehlo.add %570, %571 : tensor<2x1x1xf32>
    %573 = stablehlo.rsqrt %572 : tensor<2x1x1xf32>
    %574 = stablehlo.reshape %573 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %575 = stablehlo.broadcast_in_dim %574, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %576 = stablehlo.multiply %566, %575 : tensor<2x1x4096xf32>
    %577 = stablehlo.broadcast_in_dim %arg168, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %578 = stablehlo.multiply %576, %577 : tensor<2x1x4096xf32>
    %579 = stablehlo.reshape %578 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %580 = stablehlo.transpose %arg227, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %581 = stablehlo.dot %579, %580, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %582 = stablehlo.reshape %581 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %583 = stablehlo.logistic %582 : tensor<2x1x11008xf32>
    %584 = stablehlo.multiply %582, %583 : tensor<2x1x11008xf32>
    %585 = stablehlo.reshape %578 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %586 = stablehlo.transpose %arg167, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %587 = stablehlo.dot %585, %586, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %588 = stablehlo.reshape %587 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %589 = stablehlo.multiply %584, %588 : tensor<2x1x11008xf32>
    %590 = stablehlo.reshape %589 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %591 = stablehlo.transpose %arg166, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %592 = stablehlo.dot %590, %591, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %593 = stablehlo.reshape %592 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %594 = stablehlo.add %566, %593 : tensor<2x1x4096xf32>
    %595 = stablehlo.power %594, %1 : tensor<2x1x4096xf32>
    %596 = stablehlo.reduce(%595 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %597 = stablehlo.multiply %596, %0 : tensor<2x1xf32>
    %598 = stablehlo.reshape %597 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %599 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %600 = stablehlo.add %598, %599 : tensor<2x1x1xf32>
    %601 = stablehlo.rsqrt %600 : tensor<2x1x1xf32>
    %602 = stablehlo.reshape %601 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %603 = stablehlo.broadcast_in_dim %602, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %604 = stablehlo.multiply %594, %603 : tensor<2x1x4096xf32>
    %605 = stablehlo.broadcast_in_dim %arg165, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %606 = stablehlo.multiply %604, %605 : tensor<2x1x4096xf32>
    %607 = stablehlo.reshape %606 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %608 = stablehlo.transpose %arg231, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %609 = stablehlo.dot %607, %608, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %610 = stablehlo.reshape %609 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %611 = stablehlo.slice %610 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %612 = stablehlo.reshape %611 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %613 = stablehlo.slice %610 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %614 = stablehlo.reshape %613 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %615 = stablehlo.complex %612, %614 : tensor<2x1x32x64xcomplex<f32>>
    %616 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %617 = stablehlo.multiply %615, %616 : tensor<2x1x32x64xcomplex<f32>>
    %618 = stablehlo.real %617 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %619 = stablehlo.reshape %618 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %620 = stablehlo.imag %617 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %621 = stablehlo.reshape %620 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %622 = stablehlo.concatenate %619, %621, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %623 = stablehlo.reshape %622 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %624 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %625 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %626 = stablehlo.add %arg199, %625 : tensor<1xi64>
    %627 = stablehlo.select %624, %626, %arg199 : tensor<1xi1>, tensor<1xi64>
    %628 = stablehlo.reshape %627 : (tensor<1xi64>) -> tensor<1x1xi64>
    %629 = stablehlo.reshape %606 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %630 = stablehlo.transpose %arg229, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %631 = stablehlo.dot %629, %630, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %632 = stablehlo.reshape %631 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %633 = stablehlo.slice %632 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %634 = stablehlo.reshape %633 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %635 = stablehlo.slice %632 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %636 = stablehlo.reshape %635 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %637 = stablehlo.complex %634, %636 : tensor<2x1x32x64xcomplex<f32>>
    %638 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %639 = stablehlo.multiply %637, %638 : tensor<2x1x32x64xcomplex<f32>>
    %640 = stablehlo.real %639 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %641 = stablehlo.reshape %640 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %642 = stablehlo.imag %639 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %643 = stablehlo.reshape %642 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %644 = stablehlo.concatenate %641, %643, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %645 = stablehlo.reshape %644 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %646 = "stablehlo.scatter"(%arg230, %628, %645) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %647 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %648 = "stablehlo.gather"(%646, %647) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %649 = stablehlo.transpose %648, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %650 = stablehlo.reshape %649 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %651 = stablehlo.dot_general %623, %650, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %652 = stablehlo.reshape %651 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %653 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %654 = stablehlo.divide %652, %653 : tensor<2x32x1x2048xf32>
    %655 = stablehlo.reduce(%654 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %656 = stablehlo.broadcast_in_dim %655, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %657 = stablehlo.subtract %654, %656 : tensor<2x32x1x2048xf32>
    %658 = stablehlo.exponential %657 : tensor<2x32x1x2048xf32>
    %659 = stablehlo.reduce(%658 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %660 = stablehlo.broadcast_in_dim %659, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %661 = stablehlo.divide %658, %660 : tensor<2x32x1x2048xf32>
    %662 = stablehlo.reshape %661 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %663 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %664 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %665 = stablehlo.add %arg199, %664 : tensor<1xi64>
    %666 = stablehlo.select %663, %665, %arg199 : tensor<1xi1>, tensor<1xi64>
    %667 = stablehlo.reshape %666 : (tensor<1xi64>) -> tensor<1x1xi64>
    %668 = stablehlo.reshape %606 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %669 = stablehlo.transpose %arg164, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %670 = stablehlo.dot %668, %669, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %671 = stablehlo.reshape %670 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %672 = "stablehlo.scatter"(%arg228, %667, %671) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %673 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %674 = "stablehlo.gather"(%672, %673) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %675 = stablehlo.transpose %674, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %676 = stablehlo.reshape %675 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %677 = stablehlo.dot_general %662, %676, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %678 = stablehlo.reshape %677 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %679 = stablehlo.transpose %arg163, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %680 = stablehlo.dot %678, %679, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %681 = stablehlo.reshape %680 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %682 = stablehlo.add %594, %681 : tensor<2x1x4096xf32>
    %683 = stablehlo.power %682, %1 : tensor<2x1x4096xf32>
    %684 = stablehlo.reduce(%683 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %685 = stablehlo.multiply %684, %0 : tensor<2x1xf32>
    %686 = stablehlo.reshape %685 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %687 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %688 = stablehlo.add %686, %687 : tensor<2x1x1xf32>
    %689 = stablehlo.rsqrt %688 : tensor<2x1x1xf32>
    %690 = stablehlo.reshape %689 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %691 = stablehlo.broadcast_in_dim %690, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %692 = stablehlo.multiply %682, %691 : tensor<2x1x4096xf32>
    %693 = stablehlo.broadcast_in_dim %arg162, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %694 = stablehlo.multiply %692, %693 : tensor<2x1x4096xf32>
    %695 = stablehlo.reshape %694 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %696 = stablehlo.transpose %arg232, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %697 = stablehlo.dot %695, %696, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %698 = stablehlo.reshape %697 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %699 = stablehlo.logistic %698 : tensor<2x1x11008xf32>
    %700 = stablehlo.multiply %698, %699 : tensor<2x1x11008xf32>
    %701 = stablehlo.reshape %694 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %702 = stablehlo.transpose %arg161, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %703 = stablehlo.dot %701, %702, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %704 = stablehlo.reshape %703 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %705 = stablehlo.multiply %700, %704 : tensor<2x1x11008xf32>
    %706 = stablehlo.reshape %705 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %707 = stablehlo.transpose %arg160, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %708 = stablehlo.dot %706, %707, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %709 = stablehlo.reshape %708 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %710 = stablehlo.add %682, %709 : tensor<2x1x4096xf32>
    %711 = stablehlo.power %710, %1 : tensor<2x1x4096xf32>
    %712 = stablehlo.reduce(%711 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %713 = stablehlo.multiply %712, %0 : tensor<2x1xf32>
    %714 = stablehlo.reshape %713 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %715 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %716 = stablehlo.add %714, %715 : tensor<2x1x1xf32>
    %717 = stablehlo.rsqrt %716 : tensor<2x1x1xf32>
    %718 = stablehlo.reshape %717 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %719 = stablehlo.broadcast_in_dim %718, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %720 = stablehlo.multiply %710, %719 : tensor<2x1x4096xf32>
    %721 = stablehlo.broadcast_in_dim %arg159, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %722 = stablehlo.multiply %720, %721 : tensor<2x1x4096xf32>
    %723 = stablehlo.reshape %722 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %724 = stablehlo.transpose %arg236, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %725 = stablehlo.dot %723, %724, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %726 = stablehlo.reshape %725 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %727 = stablehlo.slice %726 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %728 = stablehlo.reshape %727 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %729 = stablehlo.slice %726 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %730 = stablehlo.reshape %729 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %731 = stablehlo.complex %728, %730 : tensor<2x1x32x64xcomplex<f32>>
    %732 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %733 = stablehlo.multiply %731, %732 : tensor<2x1x32x64xcomplex<f32>>
    %734 = stablehlo.real %733 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %735 = stablehlo.reshape %734 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %736 = stablehlo.imag %733 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %737 = stablehlo.reshape %736 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %738 = stablehlo.concatenate %735, %737, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %739 = stablehlo.reshape %738 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %740 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %741 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %742 = stablehlo.add %arg199, %741 : tensor<1xi64>
    %743 = stablehlo.select %740, %742, %arg199 : tensor<1xi1>, tensor<1xi64>
    %744 = stablehlo.reshape %743 : (tensor<1xi64>) -> tensor<1x1xi64>
    %745 = stablehlo.reshape %722 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %746 = stablehlo.transpose %arg234, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %747 = stablehlo.dot %745, %746, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %748 = stablehlo.reshape %747 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %749 = stablehlo.slice %748 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %750 = stablehlo.reshape %749 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %751 = stablehlo.slice %748 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %752 = stablehlo.reshape %751 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %753 = stablehlo.complex %750, %752 : tensor<2x1x32x64xcomplex<f32>>
    %754 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %755 = stablehlo.multiply %753, %754 : tensor<2x1x32x64xcomplex<f32>>
    %756 = stablehlo.real %755 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %757 = stablehlo.reshape %756 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %758 = stablehlo.imag %755 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %759 = stablehlo.reshape %758 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %760 = stablehlo.concatenate %757, %759, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %761 = stablehlo.reshape %760 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %762 = "stablehlo.scatter"(%arg235, %744, %761) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %763 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %764 = "stablehlo.gather"(%762, %763) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %765 = stablehlo.transpose %764, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %766 = stablehlo.reshape %765 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %767 = stablehlo.dot_general %739, %766, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %768 = stablehlo.reshape %767 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %769 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %770 = stablehlo.divide %768, %769 : tensor<2x32x1x2048xf32>
    %771 = stablehlo.reduce(%770 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %772 = stablehlo.broadcast_in_dim %771, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %773 = stablehlo.subtract %770, %772 : tensor<2x32x1x2048xf32>
    %774 = stablehlo.exponential %773 : tensor<2x32x1x2048xf32>
    %775 = stablehlo.reduce(%774 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %776 = stablehlo.broadcast_in_dim %775, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %777 = stablehlo.divide %774, %776 : tensor<2x32x1x2048xf32>
    %778 = stablehlo.reshape %777 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %779 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %780 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %781 = stablehlo.add %arg199, %780 : tensor<1xi64>
    %782 = stablehlo.select %779, %781, %arg199 : tensor<1xi1>, tensor<1xi64>
    %783 = stablehlo.reshape %782 : (tensor<1xi64>) -> tensor<1x1xi64>
    %784 = stablehlo.reshape %722 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %785 = stablehlo.transpose %arg158, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %786 = stablehlo.dot %784, %785, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %787 = stablehlo.reshape %786 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %788 = "stablehlo.scatter"(%arg233, %783, %787) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %789 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %790 = "stablehlo.gather"(%788, %789) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %791 = stablehlo.transpose %790, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %792 = stablehlo.reshape %791 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %793 = stablehlo.dot_general %778, %792, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %794 = stablehlo.reshape %793 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %795 = stablehlo.transpose %arg157, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %796 = stablehlo.dot %794, %795, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %797 = stablehlo.reshape %796 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %798 = stablehlo.add %710, %797 : tensor<2x1x4096xf32>
    %799 = stablehlo.power %798, %1 : tensor<2x1x4096xf32>
    %800 = stablehlo.reduce(%799 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %801 = stablehlo.multiply %800, %0 : tensor<2x1xf32>
    %802 = stablehlo.reshape %801 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %803 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %804 = stablehlo.add %802, %803 : tensor<2x1x1xf32>
    %805 = stablehlo.rsqrt %804 : tensor<2x1x1xf32>
    %806 = stablehlo.reshape %805 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %807 = stablehlo.broadcast_in_dim %806, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %808 = stablehlo.multiply %798, %807 : tensor<2x1x4096xf32>
    %809 = stablehlo.broadcast_in_dim %arg156, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %810 = stablehlo.multiply %808, %809 : tensor<2x1x4096xf32>
    %811 = stablehlo.reshape %810 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %812 = stablehlo.transpose %arg237, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %813 = stablehlo.dot %811, %812, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %814 = stablehlo.reshape %813 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %815 = stablehlo.logistic %814 : tensor<2x1x11008xf32>
    %816 = stablehlo.multiply %814, %815 : tensor<2x1x11008xf32>
    %817 = stablehlo.reshape %810 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %818 = stablehlo.transpose %arg155, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %819 = stablehlo.dot %817, %818, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %820 = stablehlo.reshape %819 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %821 = stablehlo.multiply %816, %820 : tensor<2x1x11008xf32>
    %822 = stablehlo.reshape %821 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %823 = stablehlo.transpose %arg154, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %824 = stablehlo.dot %822, %823, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %825 = stablehlo.reshape %824 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %826 = stablehlo.add %798, %825 : tensor<2x1x4096xf32>
    %827 = stablehlo.power %826, %1 : tensor<2x1x4096xf32>
    %828 = stablehlo.reduce(%827 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %829 = stablehlo.multiply %828, %0 : tensor<2x1xf32>
    %830 = stablehlo.reshape %829 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %831 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %832 = stablehlo.add %830, %831 : tensor<2x1x1xf32>
    %833 = stablehlo.rsqrt %832 : tensor<2x1x1xf32>
    %834 = stablehlo.reshape %833 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %835 = stablehlo.broadcast_in_dim %834, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %836 = stablehlo.multiply %826, %835 : tensor<2x1x4096xf32>
    %837 = stablehlo.broadcast_in_dim %arg153, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %838 = stablehlo.multiply %836, %837 : tensor<2x1x4096xf32>
    %839 = stablehlo.reshape %838 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %840 = stablehlo.transpose %arg241, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %841 = stablehlo.dot %839, %840, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %842 = stablehlo.reshape %841 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %843 = stablehlo.slice %842 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %844 = stablehlo.reshape %843 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %845 = stablehlo.slice %842 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %846 = stablehlo.reshape %845 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %847 = stablehlo.complex %844, %846 : tensor<2x1x32x64xcomplex<f32>>
    %848 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %849 = stablehlo.multiply %847, %848 : tensor<2x1x32x64xcomplex<f32>>
    %850 = stablehlo.real %849 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %851 = stablehlo.reshape %850 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %852 = stablehlo.imag %849 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %853 = stablehlo.reshape %852 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %854 = stablehlo.concatenate %851, %853, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %855 = stablehlo.reshape %854 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %856 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %857 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %858 = stablehlo.add %arg199, %857 : tensor<1xi64>
    %859 = stablehlo.select %856, %858, %arg199 : tensor<1xi1>, tensor<1xi64>
    %860 = stablehlo.reshape %859 : (tensor<1xi64>) -> tensor<1x1xi64>
    %861 = stablehlo.reshape %838 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %862 = stablehlo.transpose %arg239, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %863 = stablehlo.dot %861, %862, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %864 = stablehlo.reshape %863 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %865 = stablehlo.slice %864 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %866 = stablehlo.reshape %865 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %867 = stablehlo.slice %864 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %868 = stablehlo.reshape %867 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %869 = stablehlo.complex %866, %868 : tensor<2x1x32x64xcomplex<f32>>
    %870 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %871 = stablehlo.multiply %869, %870 : tensor<2x1x32x64xcomplex<f32>>
    %872 = stablehlo.real %871 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %873 = stablehlo.reshape %872 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %874 = stablehlo.imag %871 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %875 = stablehlo.reshape %874 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %876 = stablehlo.concatenate %873, %875, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %877 = stablehlo.reshape %876 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %878 = "stablehlo.scatter"(%arg240, %860, %877) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %879 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %880 = "stablehlo.gather"(%878, %879) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %881 = stablehlo.transpose %880, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %882 = stablehlo.reshape %881 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %883 = stablehlo.dot_general %855, %882, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %884 = stablehlo.reshape %883 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %885 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %886 = stablehlo.divide %884, %885 : tensor<2x32x1x2048xf32>
    %887 = stablehlo.reduce(%886 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %888 = stablehlo.broadcast_in_dim %887, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %889 = stablehlo.subtract %886, %888 : tensor<2x32x1x2048xf32>
    %890 = stablehlo.exponential %889 : tensor<2x32x1x2048xf32>
    %891 = stablehlo.reduce(%890 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %892 = stablehlo.broadcast_in_dim %891, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %893 = stablehlo.divide %890, %892 : tensor<2x32x1x2048xf32>
    %894 = stablehlo.reshape %893 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %895 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %896 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %897 = stablehlo.add %arg199, %896 : tensor<1xi64>
    %898 = stablehlo.select %895, %897, %arg199 : tensor<1xi1>, tensor<1xi64>
    %899 = stablehlo.reshape %898 : (tensor<1xi64>) -> tensor<1x1xi64>
    %900 = stablehlo.reshape %838 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %901 = stablehlo.transpose %arg152, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %902 = stablehlo.dot %900, %901, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %903 = stablehlo.reshape %902 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %904 = "stablehlo.scatter"(%arg238, %899, %903) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %905 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %906 = "stablehlo.gather"(%904, %905) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %907 = stablehlo.transpose %906, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %908 = stablehlo.reshape %907 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %909 = stablehlo.dot_general %894, %908, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %910 = stablehlo.reshape %909 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %911 = stablehlo.transpose %arg151, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %912 = stablehlo.dot %910, %911, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %913 = stablehlo.reshape %912 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %914 = stablehlo.add %826, %913 : tensor<2x1x4096xf32>
    %915 = stablehlo.power %914, %1 : tensor<2x1x4096xf32>
    %916 = stablehlo.reduce(%915 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %917 = stablehlo.multiply %916, %0 : tensor<2x1xf32>
    %918 = stablehlo.reshape %917 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %919 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %920 = stablehlo.add %918, %919 : tensor<2x1x1xf32>
    %921 = stablehlo.rsqrt %920 : tensor<2x1x1xf32>
    %922 = stablehlo.reshape %921 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %923 = stablehlo.broadcast_in_dim %922, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %924 = stablehlo.multiply %914, %923 : tensor<2x1x4096xf32>
    %925 = stablehlo.broadcast_in_dim %arg150, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %926 = stablehlo.multiply %924, %925 : tensor<2x1x4096xf32>
    %927 = stablehlo.reshape %926 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %928 = stablehlo.transpose %arg242, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %929 = stablehlo.dot %927, %928, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %930 = stablehlo.reshape %929 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %931 = stablehlo.logistic %930 : tensor<2x1x11008xf32>
    %932 = stablehlo.multiply %930, %931 : tensor<2x1x11008xf32>
    %933 = stablehlo.reshape %926 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %934 = stablehlo.transpose %arg149, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %935 = stablehlo.dot %933, %934, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %936 = stablehlo.reshape %935 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %937 = stablehlo.multiply %932, %936 : tensor<2x1x11008xf32>
    %938 = stablehlo.reshape %937 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %939 = stablehlo.transpose %arg148, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %940 = stablehlo.dot %938, %939, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %941 = stablehlo.reshape %940 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %942 = stablehlo.add %914, %941 : tensor<2x1x4096xf32>
    %943 = stablehlo.power %942, %1 : tensor<2x1x4096xf32>
    %944 = stablehlo.reduce(%943 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %945 = stablehlo.multiply %944, %0 : tensor<2x1xf32>
    %946 = stablehlo.reshape %945 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %947 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %948 = stablehlo.add %946, %947 : tensor<2x1x1xf32>
    %949 = stablehlo.rsqrt %948 : tensor<2x1x1xf32>
    %950 = stablehlo.reshape %949 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %951 = stablehlo.broadcast_in_dim %950, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %952 = stablehlo.multiply %942, %951 : tensor<2x1x4096xf32>
    %953 = stablehlo.broadcast_in_dim %arg147, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %954 = stablehlo.multiply %952, %953 : tensor<2x1x4096xf32>
    %955 = stablehlo.reshape %954 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %956 = stablehlo.transpose %arg246, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %957 = stablehlo.dot %955, %956, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %958 = stablehlo.reshape %957 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %959 = stablehlo.slice %958 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %960 = stablehlo.reshape %959 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %961 = stablehlo.slice %958 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %962 = stablehlo.reshape %961 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %963 = stablehlo.complex %960, %962 : tensor<2x1x32x64xcomplex<f32>>
    %964 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %965 = stablehlo.multiply %963, %964 : tensor<2x1x32x64xcomplex<f32>>
    %966 = stablehlo.real %965 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %967 = stablehlo.reshape %966 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %968 = stablehlo.imag %965 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %969 = stablehlo.reshape %968 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %970 = stablehlo.concatenate %967, %969, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %971 = stablehlo.reshape %970 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %972 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %973 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %974 = stablehlo.add %arg199, %973 : tensor<1xi64>
    %975 = stablehlo.select %972, %974, %arg199 : tensor<1xi1>, tensor<1xi64>
    %976 = stablehlo.reshape %975 : (tensor<1xi64>) -> tensor<1x1xi64>
    %977 = stablehlo.reshape %954 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %978 = stablehlo.transpose %arg244, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %979 = stablehlo.dot %977, %978, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %980 = stablehlo.reshape %979 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %981 = stablehlo.slice %980 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %982 = stablehlo.reshape %981 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %983 = stablehlo.slice %980 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %984 = stablehlo.reshape %983 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %985 = stablehlo.complex %982, %984 : tensor<2x1x32x64xcomplex<f32>>
    %986 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %987 = stablehlo.multiply %985, %986 : tensor<2x1x32x64xcomplex<f32>>
    %988 = stablehlo.real %987 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %989 = stablehlo.reshape %988 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %990 = stablehlo.imag %987 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %991 = stablehlo.reshape %990 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %992 = stablehlo.concatenate %989, %991, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %993 = stablehlo.reshape %992 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %994 = "stablehlo.scatter"(%arg245, %976, %993) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %995 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %996 = "stablehlo.gather"(%994, %995) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %997 = stablehlo.transpose %996, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %998 = stablehlo.reshape %997 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %999 = stablehlo.dot_general %971, %998, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %1000 = stablehlo.reshape %999 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %1001 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %1002 = stablehlo.divide %1000, %1001 : tensor<2x32x1x2048xf32>
    %1003 = stablehlo.reduce(%1002 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1004 = stablehlo.broadcast_in_dim %1003, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1005 = stablehlo.subtract %1002, %1004 : tensor<2x32x1x2048xf32>
    %1006 = stablehlo.exponential %1005 : tensor<2x32x1x2048xf32>
    %1007 = stablehlo.reduce(%1006 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1008 = stablehlo.broadcast_in_dim %1007, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1009 = stablehlo.divide %1006, %1008 : tensor<2x32x1x2048xf32>
    %1010 = stablehlo.reshape %1009 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %1011 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1012 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1013 = stablehlo.add %arg199, %1012 : tensor<1xi64>
    %1014 = stablehlo.select %1011, %1013, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1015 = stablehlo.reshape %1014 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1016 = stablehlo.reshape %954 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1017 = stablehlo.transpose %arg146, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1018 = stablehlo.dot %1016, %1017, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1019 = stablehlo.reshape %1018 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %1020 = "stablehlo.scatter"(%arg243, %1015, %1019) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1021 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1022 = "stablehlo.gather"(%1020, %1021) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1023 = stablehlo.transpose %1022, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %1024 = stablehlo.reshape %1023 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %1025 = stablehlo.dot_general %1010, %1024, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %1026 = stablehlo.reshape %1025 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %1027 = stablehlo.transpose %arg145, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1028 = stablehlo.dot %1026, %1027, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1029 = stablehlo.reshape %1028 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1030 = stablehlo.add %942, %1029 : tensor<2x1x4096xf32>
    %1031 = stablehlo.power %1030, %1 : tensor<2x1x4096xf32>
    %1032 = stablehlo.reduce(%1031 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1033 = stablehlo.multiply %1032, %0 : tensor<2x1xf32>
    %1034 = stablehlo.reshape %1033 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1035 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1036 = stablehlo.add %1034, %1035 : tensor<2x1x1xf32>
    %1037 = stablehlo.rsqrt %1036 : tensor<2x1x1xf32>
    %1038 = stablehlo.reshape %1037 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1039 = stablehlo.broadcast_in_dim %1038, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1040 = stablehlo.multiply %1030, %1039 : tensor<2x1x4096xf32>
    %1041 = stablehlo.broadcast_in_dim %arg144, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1042 = stablehlo.multiply %1040, %1041 : tensor<2x1x4096xf32>
    %1043 = stablehlo.reshape %1042 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1044 = stablehlo.transpose %arg247, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1045 = stablehlo.dot %1043, %1044, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1046 = stablehlo.reshape %1045 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1047 = stablehlo.logistic %1046 : tensor<2x1x11008xf32>
    %1048 = stablehlo.multiply %1046, %1047 : tensor<2x1x11008xf32>
    %1049 = stablehlo.reshape %1042 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1050 = stablehlo.transpose %arg143, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1051 = stablehlo.dot %1049, %1050, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1052 = stablehlo.reshape %1051 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1053 = stablehlo.multiply %1048, %1052 : tensor<2x1x11008xf32>
    %1054 = stablehlo.reshape %1053 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %1055 = stablehlo.transpose %arg142, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1056 = stablehlo.dot %1054, %1055, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %1057 = stablehlo.reshape %1056 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1058 = stablehlo.add %1030, %1057 : tensor<2x1x4096xf32>
    %1059 = stablehlo.power %1058, %1 : tensor<2x1x4096xf32>
    %1060 = stablehlo.reduce(%1059 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1061 = stablehlo.multiply %1060, %0 : tensor<2x1xf32>
    %1062 = stablehlo.reshape %1061 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1063 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1064 = stablehlo.add %1062, %1063 : tensor<2x1x1xf32>
    %1065 = stablehlo.rsqrt %1064 : tensor<2x1x1xf32>
    %1066 = stablehlo.reshape %1065 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1067 = stablehlo.broadcast_in_dim %1066, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1068 = stablehlo.multiply %1058, %1067 : tensor<2x1x4096xf32>
    %1069 = stablehlo.broadcast_in_dim %arg141, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1070 = stablehlo.multiply %1068, %1069 : tensor<2x1x4096xf32>
    %1071 = stablehlo.reshape %1070 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1072 = stablehlo.transpose %arg251, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1073 = stablehlo.dot %1071, %1072, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1074 = stablehlo.reshape %1073 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %1075 = stablehlo.slice %1074 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1076 = stablehlo.reshape %1075 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1077 = stablehlo.slice %1074 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1078 = stablehlo.reshape %1077 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1079 = stablehlo.complex %1076, %1078 : tensor<2x1x32x64xcomplex<f32>>
    %1080 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %1081 = stablehlo.multiply %1079, %1080 : tensor<2x1x32x64xcomplex<f32>>
    %1082 = stablehlo.real %1081 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1083 = stablehlo.reshape %1082 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1084 = stablehlo.imag %1081 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1085 = stablehlo.reshape %1084 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1086 = stablehlo.concatenate %1083, %1085, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %1087 = stablehlo.reshape %1086 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %1088 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1089 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1090 = stablehlo.add %arg199, %1089 : tensor<1xi64>
    %1091 = stablehlo.select %1088, %1090, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1092 = stablehlo.reshape %1091 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1093 = stablehlo.reshape %1070 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1094 = stablehlo.transpose %arg249, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1095 = stablehlo.dot %1093, %1094, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1096 = stablehlo.reshape %1095 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %1097 = stablehlo.slice %1096 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1098 = stablehlo.reshape %1097 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1099 = stablehlo.slice %1096 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1100 = stablehlo.reshape %1099 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1101 = stablehlo.complex %1098, %1100 : tensor<2x1x32x64xcomplex<f32>>
    %1102 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %1103 = stablehlo.multiply %1101, %1102 : tensor<2x1x32x64xcomplex<f32>>
    %1104 = stablehlo.real %1103 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1105 = stablehlo.reshape %1104 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1106 = stablehlo.imag %1103 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1107 = stablehlo.reshape %1106 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1108 = stablehlo.concatenate %1105, %1107, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %1109 = stablehlo.reshape %1108 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %1110 = "stablehlo.scatter"(%arg250, %1092, %1109) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1111 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1112 = "stablehlo.gather"(%1110, %1111) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1113 = stablehlo.transpose %1112, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %1114 = stablehlo.reshape %1113 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %1115 = stablehlo.dot_general %1087, %1114, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %1116 = stablehlo.reshape %1115 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %1117 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %1118 = stablehlo.divide %1116, %1117 : tensor<2x32x1x2048xf32>
    %1119 = stablehlo.reduce(%1118 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1120 = stablehlo.broadcast_in_dim %1119, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1121 = stablehlo.subtract %1118, %1120 : tensor<2x32x1x2048xf32>
    %1122 = stablehlo.exponential %1121 : tensor<2x32x1x2048xf32>
    %1123 = stablehlo.reduce(%1122 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1124 = stablehlo.broadcast_in_dim %1123, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1125 = stablehlo.divide %1122, %1124 : tensor<2x32x1x2048xf32>
    %1126 = stablehlo.reshape %1125 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %1127 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1128 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1129 = stablehlo.add %arg199, %1128 : tensor<1xi64>
    %1130 = stablehlo.select %1127, %1129, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1131 = stablehlo.reshape %1130 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1132 = stablehlo.reshape %1070 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1133 = stablehlo.transpose %arg140, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1134 = stablehlo.dot %1132, %1133, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1135 = stablehlo.reshape %1134 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %1136 = "stablehlo.scatter"(%arg248, %1131, %1135) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1137 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1138 = "stablehlo.gather"(%1136, %1137) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1139 = stablehlo.transpose %1138, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %1140 = stablehlo.reshape %1139 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %1141 = stablehlo.dot_general %1126, %1140, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %1142 = stablehlo.reshape %1141 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %1143 = stablehlo.transpose %arg139, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1144 = stablehlo.dot %1142, %1143, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1145 = stablehlo.reshape %1144 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1146 = stablehlo.add %1058, %1145 : tensor<2x1x4096xf32>
    %1147 = stablehlo.power %1146, %1 : tensor<2x1x4096xf32>
    %1148 = stablehlo.reduce(%1147 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1149 = stablehlo.multiply %1148, %0 : tensor<2x1xf32>
    %1150 = stablehlo.reshape %1149 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1151 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1152 = stablehlo.add %1150, %1151 : tensor<2x1x1xf32>
    %1153 = stablehlo.rsqrt %1152 : tensor<2x1x1xf32>
    %1154 = stablehlo.reshape %1153 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1155 = stablehlo.broadcast_in_dim %1154, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1156 = stablehlo.multiply %1146, %1155 : tensor<2x1x4096xf32>
    %1157 = stablehlo.broadcast_in_dim %arg138, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1158 = stablehlo.multiply %1156, %1157 : tensor<2x1x4096xf32>
    %1159 = stablehlo.reshape %1158 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1160 = stablehlo.transpose %arg252, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1161 = stablehlo.dot %1159, %1160, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1162 = stablehlo.reshape %1161 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1163 = stablehlo.logistic %1162 : tensor<2x1x11008xf32>
    %1164 = stablehlo.multiply %1162, %1163 : tensor<2x1x11008xf32>
    %1165 = stablehlo.reshape %1158 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1166 = stablehlo.transpose %arg137, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1167 = stablehlo.dot %1165, %1166, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1168 = stablehlo.reshape %1167 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1169 = stablehlo.multiply %1164, %1168 : tensor<2x1x11008xf32>
    %1170 = stablehlo.reshape %1169 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %1171 = stablehlo.transpose %arg136, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1172 = stablehlo.dot %1170, %1171, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %1173 = stablehlo.reshape %1172 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1174 = stablehlo.add %1146, %1173 : tensor<2x1x4096xf32>
    %1175 = stablehlo.power %1174, %1 : tensor<2x1x4096xf32>
    %1176 = stablehlo.reduce(%1175 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1177 = stablehlo.multiply %1176, %0 : tensor<2x1xf32>
    %1178 = stablehlo.reshape %1177 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1179 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1180 = stablehlo.add %1178, %1179 : tensor<2x1x1xf32>
    %1181 = stablehlo.rsqrt %1180 : tensor<2x1x1xf32>
    %1182 = stablehlo.reshape %1181 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1183 = stablehlo.broadcast_in_dim %1182, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1184 = stablehlo.multiply %1174, %1183 : tensor<2x1x4096xf32>
    %1185 = stablehlo.broadcast_in_dim %arg135, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1186 = stablehlo.multiply %1184, %1185 : tensor<2x1x4096xf32>
    %1187 = stablehlo.reshape %1186 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1188 = stablehlo.transpose %arg256, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1189 = stablehlo.dot %1187, %1188, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1190 = stablehlo.reshape %1189 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %1191 = stablehlo.slice %1190 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1192 = stablehlo.reshape %1191 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1193 = stablehlo.slice %1190 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1194 = stablehlo.reshape %1193 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1195 = stablehlo.complex %1192, %1194 : tensor<2x1x32x64xcomplex<f32>>
    %1196 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %1197 = stablehlo.multiply %1195, %1196 : tensor<2x1x32x64xcomplex<f32>>
    %1198 = stablehlo.real %1197 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1199 = stablehlo.reshape %1198 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1200 = stablehlo.imag %1197 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1201 = stablehlo.reshape %1200 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1202 = stablehlo.concatenate %1199, %1201, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %1203 = stablehlo.reshape %1202 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %1204 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1205 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1206 = stablehlo.add %arg199, %1205 : tensor<1xi64>
    %1207 = stablehlo.select %1204, %1206, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1208 = stablehlo.reshape %1207 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1209 = stablehlo.reshape %1186 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1210 = stablehlo.transpose %arg254, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1211 = stablehlo.dot %1209, %1210, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1212 = stablehlo.reshape %1211 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %1213 = stablehlo.slice %1212 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1214 = stablehlo.reshape %1213 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1215 = stablehlo.slice %1212 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1216 = stablehlo.reshape %1215 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1217 = stablehlo.complex %1214, %1216 : tensor<2x1x32x64xcomplex<f32>>
    %1218 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %1219 = stablehlo.multiply %1217, %1218 : tensor<2x1x32x64xcomplex<f32>>
    %1220 = stablehlo.real %1219 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1221 = stablehlo.reshape %1220 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1222 = stablehlo.imag %1219 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1223 = stablehlo.reshape %1222 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1224 = stablehlo.concatenate %1221, %1223, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %1225 = stablehlo.reshape %1224 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %1226 = "stablehlo.scatter"(%arg255, %1208, %1225) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1227 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1228 = "stablehlo.gather"(%1226, %1227) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1229 = stablehlo.transpose %1228, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %1230 = stablehlo.reshape %1229 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %1231 = stablehlo.dot_general %1203, %1230, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %1232 = stablehlo.reshape %1231 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %1233 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %1234 = stablehlo.divide %1232, %1233 : tensor<2x32x1x2048xf32>
    %1235 = stablehlo.reduce(%1234 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1236 = stablehlo.broadcast_in_dim %1235, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1237 = stablehlo.subtract %1234, %1236 : tensor<2x32x1x2048xf32>
    %1238 = stablehlo.exponential %1237 : tensor<2x32x1x2048xf32>
    %1239 = stablehlo.reduce(%1238 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1240 = stablehlo.broadcast_in_dim %1239, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1241 = stablehlo.divide %1238, %1240 : tensor<2x32x1x2048xf32>
    %1242 = stablehlo.reshape %1241 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %1243 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1244 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1245 = stablehlo.add %arg199, %1244 : tensor<1xi64>
    %1246 = stablehlo.select %1243, %1245, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1247 = stablehlo.reshape %1246 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1248 = stablehlo.reshape %1186 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1249 = stablehlo.transpose %arg134, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1250 = stablehlo.dot %1248, %1249, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1251 = stablehlo.reshape %1250 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %1252 = "stablehlo.scatter"(%arg253, %1247, %1251) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1253 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1254 = "stablehlo.gather"(%1252, %1253) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1255 = stablehlo.transpose %1254, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %1256 = stablehlo.reshape %1255 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %1257 = stablehlo.dot_general %1242, %1256, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %1258 = stablehlo.reshape %1257 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %1259 = stablehlo.transpose %arg133, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1260 = stablehlo.dot %1258, %1259, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1261 = stablehlo.reshape %1260 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1262 = stablehlo.add %1174, %1261 : tensor<2x1x4096xf32>
    %1263 = stablehlo.power %1262, %1 : tensor<2x1x4096xf32>
    %1264 = stablehlo.reduce(%1263 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1265 = stablehlo.multiply %1264, %0 : tensor<2x1xf32>
    %1266 = stablehlo.reshape %1265 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1267 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1268 = stablehlo.add %1266, %1267 : tensor<2x1x1xf32>
    %1269 = stablehlo.rsqrt %1268 : tensor<2x1x1xf32>
    %1270 = stablehlo.reshape %1269 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1271 = stablehlo.broadcast_in_dim %1270, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1272 = stablehlo.multiply %1262, %1271 : tensor<2x1x4096xf32>
    %1273 = stablehlo.broadcast_in_dim %arg132, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1274 = stablehlo.multiply %1272, %1273 : tensor<2x1x4096xf32>
    %1275 = stablehlo.reshape %1274 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1276 = stablehlo.transpose %arg257, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1277 = stablehlo.dot %1275, %1276, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1278 = stablehlo.reshape %1277 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1279 = stablehlo.logistic %1278 : tensor<2x1x11008xf32>
    %1280 = stablehlo.multiply %1278, %1279 : tensor<2x1x11008xf32>
    %1281 = stablehlo.reshape %1274 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1282 = stablehlo.transpose %arg131, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1283 = stablehlo.dot %1281, %1282, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1284 = stablehlo.reshape %1283 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1285 = stablehlo.multiply %1280, %1284 : tensor<2x1x11008xf32>
    %1286 = stablehlo.reshape %1285 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %1287 = stablehlo.transpose %arg130, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1288 = stablehlo.dot %1286, %1287, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %1289 = stablehlo.reshape %1288 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1290 = stablehlo.add %1262, %1289 : tensor<2x1x4096xf32>
    %1291 = stablehlo.power %1290, %1 : tensor<2x1x4096xf32>
    %1292 = stablehlo.reduce(%1291 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1293 = stablehlo.multiply %1292, %0 : tensor<2x1xf32>
    %1294 = stablehlo.reshape %1293 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1295 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1296 = stablehlo.add %1294, %1295 : tensor<2x1x1xf32>
    %1297 = stablehlo.rsqrt %1296 : tensor<2x1x1xf32>
    %1298 = stablehlo.reshape %1297 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1299 = stablehlo.broadcast_in_dim %1298, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1300 = stablehlo.multiply %1290, %1299 : tensor<2x1x4096xf32>
    %1301 = stablehlo.broadcast_in_dim %arg129, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1302 = stablehlo.multiply %1300, %1301 : tensor<2x1x4096xf32>
    %1303 = stablehlo.reshape %1302 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1304 = stablehlo.transpose %arg261, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1305 = stablehlo.dot %1303, %1304, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1306 = stablehlo.reshape %1305 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %1307 = stablehlo.slice %1306 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1308 = stablehlo.reshape %1307 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1309 = stablehlo.slice %1306 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1310 = stablehlo.reshape %1309 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1311 = stablehlo.complex %1308, %1310 : tensor<2x1x32x64xcomplex<f32>>
    %1312 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %1313 = stablehlo.multiply %1311, %1312 : tensor<2x1x32x64xcomplex<f32>>
    %1314 = stablehlo.real %1313 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1315 = stablehlo.reshape %1314 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1316 = stablehlo.imag %1313 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1317 = stablehlo.reshape %1316 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1318 = stablehlo.concatenate %1315, %1317, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %1319 = stablehlo.reshape %1318 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %1320 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1321 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1322 = stablehlo.add %arg199, %1321 : tensor<1xi64>
    %1323 = stablehlo.select %1320, %1322, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1324 = stablehlo.reshape %1323 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1325 = stablehlo.reshape %1302 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1326 = stablehlo.transpose %arg259, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1327 = stablehlo.dot %1325, %1326, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1328 = stablehlo.reshape %1327 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %1329 = stablehlo.slice %1328 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1330 = stablehlo.reshape %1329 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1331 = stablehlo.slice %1328 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1332 = stablehlo.reshape %1331 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1333 = stablehlo.complex %1330, %1332 : tensor<2x1x32x64xcomplex<f32>>
    %1334 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %1335 = stablehlo.multiply %1333, %1334 : tensor<2x1x32x64xcomplex<f32>>
    %1336 = stablehlo.real %1335 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1337 = stablehlo.reshape %1336 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1338 = stablehlo.imag %1335 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1339 = stablehlo.reshape %1338 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1340 = stablehlo.concatenate %1337, %1339, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %1341 = stablehlo.reshape %1340 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %1342 = "stablehlo.scatter"(%arg260, %1324, %1341) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1343 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1344 = "stablehlo.gather"(%1342, %1343) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1345 = stablehlo.transpose %1344, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %1346 = stablehlo.reshape %1345 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %1347 = stablehlo.dot_general %1319, %1346, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %1348 = stablehlo.reshape %1347 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %1349 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %1350 = stablehlo.divide %1348, %1349 : tensor<2x32x1x2048xf32>
    %1351 = stablehlo.reduce(%1350 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1352 = stablehlo.broadcast_in_dim %1351, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1353 = stablehlo.subtract %1350, %1352 : tensor<2x32x1x2048xf32>
    %1354 = stablehlo.exponential %1353 : tensor<2x32x1x2048xf32>
    %1355 = stablehlo.reduce(%1354 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1356 = stablehlo.broadcast_in_dim %1355, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1357 = stablehlo.divide %1354, %1356 : tensor<2x32x1x2048xf32>
    %1358 = stablehlo.reshape %1357 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %1359 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1360 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1361 = stablehlo.add %arg199, %1360 : tensor<1xi64>
    %1362 = stablehlo.select %1359, %1361, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1363 = stablehlo.reshape %1362 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1364 = stablehlo.reshape %1302 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1365 = stablehlo.transpose %arg128, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1366 = stablehlo.dot %1364, %1365, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1367 = stablehlo.reshape %1366 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %1368 = "stablehlo.scatter"(%arg258, %1363, %1367) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1369 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1370 = "stablehlo.gather"(%1368, %1369) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1371 = stablehlo.transpose %1370, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %1372 = stablehlo.reshape %1371 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %1373 = stablehlo.dot_general %1358, %1372, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %1374 = stablehlo.reshape %1373 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %1375 = stablehlo.transpose %arg127, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1376 = stablehlo.dot %1374, %1375, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1377 = stablehlo.reshape %1376 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1378 = stablehlo.add %1290, %1377 : tensor<2x1x4096xf32>
    %1379 = stablehlo.power %1378, %1 : tensor<2x1x4096xf32>
    %1380 = stablehlo.reduce(%1379 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1381 = stablehlo.multiply %1380, %0 : tensor<2x1xf32>
    %1382 = stablehlo.reshape %1381 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1383 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1384 = stablehlo.add %1382, %1383 : tensor<2x1x1xf32>
    %1385 = stablehlo.rsqrt %1384 : tensor<2x1x1xf32>
    %1386 = stablehlo.reshape %1385 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1387 = stablehlo.broadcast_in_dim %1386, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1388 = stablehlo.multiply %1378, %1387 : tensor<2x1x4096xf32>
    %1389 = stablehlo.broadcast_in_dim %arg126, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1390 = stablehlo.multiply %1388, %1389 : tensor<2x1x4096xf32>
    %1391 = stablehlo.reshape %1390 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1392 = stablehlo.transpose %arg262, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1393 = stablehlo.dot %1391, %1392, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1394 = stablehlo.reshape %1393 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1395 = stablehlo.logistic %1394 : tensor<2x1x11008xf32>
    %1396 = stablehlo.multiply %1394, %1395 : tensor<2x1x11008xf32>
    %1397 = stablehlo.reshape %1390 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1398 = stablehlo.transpose %arg125, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1399 = stablehlo.dot %1397, %1398, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1400 = stablehlo.reshape %1399 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1401 = stablehlo.multiply %1396, %1400 : tensor<2x1x11008xf32>
    %1402 = stablehlo.reshape %1401 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %1403 = stablehlo.transpose %arg124, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1404 = stablehlo.dot %1402, %1403, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %1405 = stablehlo.reshape %1404 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1406 = stablehlo.add %1378, %1405 : tensor<2x1x4096xf32>
    %1407 = stablehlo.power %1406, %1 : tensor<2x1x4096xf32>
    %1408 = stablehlo.reduce(%1407 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1409 = stablehlo.multiply %1408, %0 : tensor<2x1xf32>
    %1410 = stablehlo.reshape %1409 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1411 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1412 = stablehlo.add %1410, %1411 : tensor<2x1x1xf32>
    %1413 = stablehlo.rsqrt %1412 : tensor<2x1x1xf32>
    %1414 = stablehlo.reshape %1413 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1415 = stablehlo.broadcast_in_dim %1414, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1416 = stablehlo.multiply %1406, %1415 : tensor<2x1x4096xf32>
    %1417 = stablehlo.broadcast_in_dim %arg123, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1418 = stablehlo.multiply %1416, %1417 : tensor<2x1x4096xf32>
    %1419 = stablehlo.reshape %1418 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1420 = stablehlo.transpose %arg266, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1421 = stablehlo.dot %1419, %1420, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1422 = stablehlo.reshape %1421 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %1423 = stablehlo.slice %1422 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1424 = stablehlo.reshape %1423 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1425 = stablehlo.slice %1422 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1426 = stablehlo.reshape %1425 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1427 = stablehlo.complex %1424, %1426 : tensor<2x1x32x64xcomplex<f32>>
    %1428 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %1429 = stablehlo.multiply %1427, %1428 : tensor<2x1x32x64xcomplex<f32>>
    %1430 = stablehlo.real %1429 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1431 = stablehlo.reshape %1430 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1432 = stablehlo.imag %1429 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1433 = stablehlo.reshape %1432 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1434 = stablehlo.concatenate %1431, %1433, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %1435 = stablehlo.reshape %1434 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %1436 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1437 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1438 = stablehlo.add %arg199, %1437 : tensor<1xi64>
    %1439 = stablehlo.select %1436, %1438, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1440 = stablehlo.reshape %1439 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1441 = stablehlo.reshape %1418 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1442 = stablehlo.transpose %arg264, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1443 = stablehlo.dot %1441, %1442, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1444 = stablehlo.reshape %1443 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %1445 = stablehlo.slice %1444 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1446 = stablehlo.reshape %1445 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1447 = stablehlo.slice %1444 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1448 = stablehlo.reshape %1447 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1449 = stablehlo.complex %1446, %1448 : tensor<2x1x32x64xcomplex<f32>>
    %1450 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %1451 = stablehlo.multiply %1449, %1450 : tensor<2x1x32x64xcomplex<f32>>
    %1452 = stablehlo.real %1451 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1453 = stablehlo.reshape %1452 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1454 = stablehlo.imag %1451 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1455 = stablehlo.reshape %1454 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1456 = stablehlo.concatenate %1453, %1455, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %1457 = stablehlo.reshape %1456 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %1458 = "stablehlo.scatter"(%arg265, %1440, %1457) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1459 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1460 = "stablehlo.gather"(%1458, %1459) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1461 = stablehlo.transpose %1460, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %1462 = stablehlo.reshape %1461 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %1463 = stablehlo.dot_general %1435, %1462, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %1464 = stablehlo.reshape %1463 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %1465 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %1466 = stablehlo.divide %1464, %1465 : tensor<2x32x1x2048xf32>
    %1467 = stablehlo.reduce(%1466 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1468 = stablehlo.broadcast_in_dim %1467, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1469 = stablehlo.subtract %1466, %1468 : tensor<2x32x1x2048xf32>
    %1470 = stablehlo.exponential %1469 : tensor<2x32x1x2048xf32>
    %1471 = stablehlo.reduce(%1470 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1472 = stablehlo.broadcast_in_dim %1471, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1473 = stablehlo.divide %1470, %1472 : tensor<2x32x1x2048xf32>
    %1474 = stablehlo.reshape %1473 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %1475 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1476 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1477 = stablehlo.add %arg199, %1476 : tensor<1xi64>
    %1478 = stablehlo.select %1475, %1477, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1479 = stablehlo.reshape %1478 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1480 = stablehlo.reshape %1418 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1481 = stablehlo.transpose %arg122, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1482 = stablehlo.dot %1480, %1481, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1483 = stablehlo.reshape %1482 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %1484 = "stablehlo.scatter"(%arg263, %1479, %1483) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1485 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1486 = "stablehlo.gather"(%1484, %1485) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1487 = stablehlo.transpose %1486, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %1488 = stablehlo.reshape %1487 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %1489 = stablehlo.dot_general %1474, %1488, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %1490 = stablehlo.reshape %1489 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %1491 = stablehlo.transpose %arg121, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1492 = stablehlo.dot %1490, %1491, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1493 = stablehlo.reshape %1492 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1494 = stablehlo.add %1406, %1493 : tensor<2x1x4096xf32>
    %1495 = stablehlo.power %1494, %1 : tensor<2x1x4096xf32>
    %1496 = stablehlo.reduce(%1495 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1497 = stablehlo.multiply %1496, %0 : tensor<2x1xf32>
    %1498 = stablehlo.reshape %1497 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1499 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1500 = stablehlo.add %1498, %1499 : tensor<2x1x1xf32>
    %1501 = stablehlo.rsqrt %1500 : tensor<2x1x1xf32>
    %1502 = stablehlo.reshape %1501 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1503 = stablehlo.broadcast_in_dim %1502, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1504 = stablehlo.multiply %1494, %1503 : tensor<2x1x4096xf32>
    %1505 = stablehlo.broadcast_in_dim %arg120, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1506 = stablehlo.multiply %1504, %1505 : tensor<2x1x4096xf32>
    %1507 = stablehlo.reshape %1506 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1508 = stablehlo.transpose %arg267, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1509 = stablehlo.dot %1507, %1508, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1510 = stablehlo.reshape %1509 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1511 = stablehlo.logistic %1510 : tensor<2x1x11008xf32>
    %1512 = stablehlo.multiply %1510, %1511 : tensor<2x1x11008xf32>
    %1513 = stablehlo.reshape %1506 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1514 = stablehlo.transpose %arg119, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1515 = stablehlo.dot %1513, %1514, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1516 = stablehlo.reshape %1515 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1517 = stablehlo.multiply %1512, %1516 : tensor<2x1x11008xf32>
    %1518 = stablehlo.reshape %1517 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %1519 = stablehlo.transpose %arg118, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1520 = stablehlo.dot %1518, %1519, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %1521 = stablehlo.reshape %1520 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1522 = stablehlo.add %1494, %1521 : tensor<2x1x4096xf32>
    %1523 = stablehlo.power %1522, %1 : tensor<2x1x4096xf32>
    %1524 = stablehlo.reduce(%1523 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1525 = stablehlo.multiply %1524, %0 : tensor<2x1xf32>
    %1526 = stablehlo.reshape %1525 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1527 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1528 = stablehlo.add %1526, %1527 : tensor<2x1x1xf32>
    %1529 = stablehlo.rsqrt %1528 : tensor<2x1x1xf32>
    %1530 = stablehlo.reshape %1529 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1531 = stablehlo.broadcast_in_dim %1530, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1532 = stablehlo.multiply %1522, %1531 : tensor<2x1x4096xf32>
    %1533 = stablehlo.broadcast_in_dim %arg117, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1534 = stablehlo.multiply %1532, %1533 : tensor<2x1x4096xf32>
    %1535 = stablehlo.reshape %1534 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1536 = stablehlo.transpose %arg271, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1537 = stablehlo.dot %1535, %1536, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1538 = stablehlo.reshape %1537 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %1539 = stablehlo.slice %1538 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1540 = stablehlo.reshape %1539 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1541 = stablehlo.slice %1538 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1542 = stablehlo.reshape %1541 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1543 = stablehlo.complex %1540, %1542 : tensor<2x1x32x64xcomplex<f32>>
    %1544 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %1545 = stablehlo.multiply %1543, %1544 : tensor<2x1x32x64xcomplex<f32>>
    %1546 = stablehlo.real %1545 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1547 = stablehlo.reshape %1546 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1548 = stablehlo.imag %1545 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1549 = stablehlo.reshape %1548 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1550 = stablehlo.concatenate %1547, %1549, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %1551 = stablehlo.reshape %1550 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %1552 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1553 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1554 = stablehlo.add %arg199, %1553 : tensor<1xi64>
    %1555 = stablehlo.select %1552, %1554, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1556 = stablehlo.reshape %1555 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1557 = stablehlo.reshape %1534 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1558 = stablehlo.transpose %arg269, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1559 = stablehlo.dot %1557, %1558, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1560 = stablehlo.reshape %1559 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %1561 = stablehlo.slice %1560 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1562 = stablehlo.reshape %1561 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1563 = stablehlo.slice %1560 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1564 = stablehlo.reshape %1563 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1565 = stablehlo.complex %1562, %1564 : tensor<2x1x32x64xcomplex<f32>>
    %1566 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %1567 = stablehlo.multiply %1565, %1566 : tensor<2x1x32x64xcomplex<f32>>
    %1568 = stablehlo.real %1567 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1569 = stablehlo.reshape %1568 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1570 = stablehlo.imag %1567 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1571 = stablehlo.reshape %1570 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1572 = stablehlo.concatenate %1569, %1571, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %1573 = stablehlo.reshape %1572 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %1574 = "stablehlo.scatter"(%arg270, %1556, %1573) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1575 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1576 = "stablehlo.gather"(%1574, %1575) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1577 = stablehlo.transpose %1576, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %1578 = stablehlo.reshape %1577 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %1579 = stablehlo.dot_general %1551, %1578, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %1580 = stablehlo.reshape %1579 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %1581 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %1582 = stablehlo.divide %1580, %1581 : tensor<2x32x1x2048xf32>
    %1583 = stablehlo.reduce(%1582 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1584 = stablehlo.broadcast_in_dim %1583, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1585 = stablehlo.subtract %1582, %1584 : tensor<2x32x1x2048xf32>
    %1586 = stablehlo.exponential %1585 : tensor<2x32x1x2048xf32>
    %1587 = stablehlo.reduce(%1586 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1588 = stablehlo.broadcast_in_dim %1587, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1589 = stablehlo.divide %1586, %1588 : tensor<2x32x1x2048xf32>
    %1590 = stablehlo.reshape %1589 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %1591 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1592 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1593 = stablehlo.add %arg199, %1592 : tensor<1xi64>
    %1594 = stablehlo.select %1591, %1593, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1595 = stablehlo.reshape %1594 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1596 = stablehlo.reshape %1534 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1597 = stablehlo.transpose %arg116, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1598 = stablehlo.dot %1596, %1597, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1599 = stablehlo.reshape %1598 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %1600 = "stablehlo.scatter"(%arg268, %1595, %1599) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1601 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1602 = "stablehlo.gather"(%1600, %1601) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1603 = stablehlo.transpose %1602, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %1604 = stablehlo.reshape %1603 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %1605 = stablehlo.dot_general %1590, %1604, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %1606 = stablehlo.reshape %1605 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %1607 = stablehlo.transpose %arg115, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1608 = stablehlo.dot %1606, %1607, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1609 = stablehlo.reshape %1608 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1610 = stablehlo.add %1522, %1609 : tensor<2x1x4096xf32>
    %1611 = stablehlo.power %1610, %1 : tensor<2x1x4096xf32>
    %1612 = stablehlo.reduce(%1611 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1613 = stablehlo.multiply %1612, %0 : tensor<2x1xf32>
    %1614 = stablehlo.reshape %1613 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1615 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1616 = stablehlo.add %1614, %1615 : tensor<2x1x1xf32>
    %1617 = stablehlo.rsqrt %1616 : tensor<2x1x1xf32>
    %1618 = stablehlo.reshape %1617 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1619 = stablehlo.broadcast_in_dim %1618, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1620 = stablehlo.multiply %1610, %1619 : tensor<2x1x4096xf32>
    %1621 = stablehlo.broadcast_in_dim %arg114, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1622 = stablehlo.multiply %1620, %1621 : tensor<2x1x4096xf32>
    %1623 = stablehlo.reshape %1622 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1624 = stablehlo.transpose %arg272, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1625 = stablehlo.dot %1623, %1624, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1626 = stablehlo.reshape %1625 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1627 = stablehlo.logistic %1626 : tensor<2x1x11008xf32>
    %1628 = stablehlo.multiply %1626, %1627 : tensor<2x1x11008xf32>
    %1629 = stablehlo.reshape %1622 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1630 = stablehlo.transpose %arg113, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1631 = stablehlo.dot %1629, %1630, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1632 = stablehlo.reshape %1631 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1633 = stablehlo.multiply %1628, %1632 : tensor<2x1x11008xf32>
    %1634 = stablehlo.reshape %1633 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %1635 = stablehlo.transpose %arg112, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1636 = stablehlo.dot %1634, %1635, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %1637 = stablehlo.reshape %1636 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1638 = stablehlo.add %1610, %1637 : tensor<2x1x4096xf32>
    %1639 = stablehlo.power %1638, %1 : tensor<2x1x4096xf32>
    %1640 = stablehlo.reduce(%1639 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1641 = stablehlo.multiply %1640, %0 : tensor<2x1xf32>
    %1642 = stablehlo.reshape %1641 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1643 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1644 = stablehlo.add %1642, %1643 : tensor<2x1x1xf32>
    %1645 = stablehlo.rsqrt %1644 : tensor<2x1x1xf32>
    %1646 = stablehlo.reshape %1645 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1647 = stablehlo.broadcast_in_dim %1646, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1648 = stablehlo.multiply %1638, %1647 : tensor<2x1x4096xf32>
    %1649 = stablehlo.broadcast_in_dim %arg111, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1650 = stablehlo.multiply %1648, %1649 : tensor<2x1x4096xf32>
    %1651 = stablehlo.reshape %1650 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1652 = stablehlo.transpose %arg276, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1653 = stablehlo.dot %1651, %1652, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1654 = stablehlo.reshape %1653 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %1655 = stablehlo.slice %1654 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1656 = stablehlo.reshape %1655 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1657 = stablehlo.slice %1654 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1658 = stablehlo.reshape %1657 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1659 = stablehlo.complex %1656, %1658 : tensor<2x1x32x64xcomplex<f32>>
    %1660 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %1661 = stablehlo.multiply %1659, %1660 : tensor<2x1x32x64xcomplex<f32>>
    %1662 = stablehlo.real %1661 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1663 = stablehlo.reshape %1662 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1664 = stablehlo.imag %1661 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1665 = stablehlo.reshape %1664 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1666 = stablehlo.concatenate %1663, %1665, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %1667 = stablehlo.reshape %1666 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %1668 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1669 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1670 = stablehlo.add %arg199, %1669 : tensor<1xi64>
    %1671 = stablehlo.select %1668, %1670, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1672 = stablehlo.reshape %1671 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1673 = stablehlo.reshape %1650 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1674 = stablehlo.transpose %arg274, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1675 = stablehlo.dot %1673, %1674, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1676 = stablehlo.reshape %1675 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %1677 = stablehlo.slice %1676 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1678 = stablehlo.reshape %1677 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1679 = stablehlo.slice %1676 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1680 = stablehlo.reshape %1679 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1681 = stablehlo.complex %1678, %1680 : tensor<2x1x32x64xcomplex<f32>>
    %1682 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %1683 = stablehlo.multiply %1681, %1682 : tensor<2x1x32x64xcomplex<f32>>
    %1684 = stablehlo.real %1683 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1685 = stablehlo.reshape %1684 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1686 = stablehlo.imag %1683 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1687 = stablehlo.reshape %1686 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1688 = stablehlo.concatenate %1685, %1687, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %1689 = stablehlo.reshape %1688 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %1690 = "stablehlo.scatter"(%arg275, %1672, %1689) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1691 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1692 = "stablehlo.gather"(%1690, %1691) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1693 = stablehlo.transpose %1692, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %1694 = stablehlo.reshape %1693 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %1695 = stablehlo.dot_general %1667, %1694, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %1696 = stablehlo.reshape %1695 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %1697 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %1698 = stablehlo.divide %1696, %1697 : tensor<2x32x1x2048xf32>
    %1699 = stablehlo.reduce(%1698 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1700 = stablehlo.broadcast_in_dim %1699, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1701 = stablehlo.subtract %1698, %1700 : tensor<2x32x1x2048xf32>
    %1702 = stablehlo.exponential %1701 : tensor<2x32x1x2048xf32>
    %1703 = stablehlo.reduce(%1702 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1704 = stablehlo.broadcast_in_dim %1703, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1705 = stablehlo.divide %1702, %1704 : tensor<2x32x1x2048xf32>
    %1706 = stablehlo.reshape %1705 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %1707 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1708 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1709 = stablehlo.add %arg199, %1708 : tensor<1xi64>
    %1710 = stablehlo.select %1707, %1709, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1711 = stablehlo.reshape %1710 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1712 = stablehlo.reshape %1650 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1713 = stablehlo.transpose %arg110, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1714 = stablehlo.dot %1712, %1713, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1715 = stablehlo.reshape %1714 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %1716 = "stablehlo.scatter"(%arg273, %1711, %1715) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1717 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1718 = "stablehlo.gather"(%1716, %1717) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1719 = stablehlo.transpose %1718, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %1720 = stablehlo.reshape %1719 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %1721 = stablehlo.dot_general %1706, %1720, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %1722 = stablehlo.reshape %1721 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %1723 = stablehlo.transpose %arg109, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1724 = stablehlo.dot %1722, %1723, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1725 = stablehlo.reshape %1724 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1726 = stablehlo.add %1638, %1725 : tensor<2x1x4096xf32>
    %1727 = stablehlo.power %1726, %1 : tensor<2x1x4096xf32>
    %1728 = stablehlo.reduce(%1727 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1729 = stablehlo.multiply %1728, %0 : tensor<2x1xf32>
    %1730 = stablehlo.reshape %1729 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1731 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1732 = stablehlo.add %1730, %1731 : tensor<2x1x1xf32>
    %1733 = stablehlo.rsqrt %1732 : tensor<2x1x1xf32>
    %1734 = stablehlo.reshape %1733 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1735 = stablehlo.broadcast_in_dim %1734, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1736 = stablehlo.multiply %1726, %1735 : tensor<2x1x4096xf32>
    %1737 = stablehlo.broadcast_in_dim %arg108, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1738 = stablehlo.multiply %1736, %1737 : tensor<2x1x4096xf32>
    %1739 = stablehlo.reshape %1738 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1740 = stablehlo.transpose %arg277, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1741 = stablehlo.dot %1739, %1740, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1742 = stablehlo.reshape %1741 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1743 = stablehlo.logistic %1742 : tensor<2x1x11008xf32>
    %1744 = stablehlo.multiply %1742, %1743 : tensor<2x1x11008xf32>
    %1745 = stablehlo.reshape %1738 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1746 = stablehlo.transpose %arg107, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1747 = stablehlo.dot %1745, %1746, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1748 = stablehlo.reshape %1747 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1749 = stablehlo.multiply %1744, %1748 : tensor<2x1x11008xf32>
    %1750 = stablehlo.reshape %1749 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %1751 = stablehlo.transpose %arg106, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1752 = stablehlo.dot %1750, %1751, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %1753 = stablehlo.reshape %1752 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1754 = stablehlo.add %1726, %1753 : tensor<2x1x4096xf32>
    %1755 = stablehlo.power %1754, %1 : tensor<2x1x4096xf32>
    %1756 = stablehlo.reduce(%1755 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1757 = stablehlo.multiply %1756, %0 : tensor<2x1xf32>
    %1758 = stablehlo.reshape %1757 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1759 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1760 = stablehlo.add %1758, %1759 : tensor<2x1x1xf32>
    %1761 = stablehlo.rsqrt %1760 : tensor<2x1x1xf32>
    %1762 = stablehlo.reshape %1761 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1763 = stablehlo.broadcast_in_dim %1762, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1764 = stablehlo.multiply %1754, %1763 : tensor<2x1x4096xf32>
    %1765 = stablehlo.broadcast_in_dim %arg105, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1766 = stablehlo.multiply %1764, %1765 : tensor<2x1x4096xf32>
    %1767 = stablehlo.reshape %1766 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1768 = stablehlo.transpose %arg281, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1769 = stablehlo.dot %1767, %1768, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1770 = stablehlo.reshape %1769 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %1771 = stablehlo.slice %1770 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1772 = stablehlo.reshape %1771 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1773 = stablehlo.slice %1770 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1774 = stablehlo.reshape %1773 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1775 = stablehlo.complex %1772, %1774 : tensor<2x1x32x64xcomplex<f32>>
    %1776 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %1777 = stablehlo.multiply %1775, %1776 : tensor<2x1x32x64xcomplex<f32>>
    %1778 = stablehlo.real %1777 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1779 = stablehlo.reshape %1778 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1780 = stablehlo.imag %1777 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1781 = stablehlo.reshape %1780 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1782 = stablehlo.concatenate %1779, %1781, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %1783 = stablehlo.reshape %1782 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %1784 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1785 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1786 = stablehlo.add %arg199, %1785 : tensor<1xi64>
    %1787 = stablehlo.select %1784, %1786, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1788 = stablehlo.reshape %1787 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1789 = stablehlo.reshape %1766 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1790 = stablehlo.transpose %arg279, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1791 = stablehlo.dot %1789, %1790, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1792 = stablehlo.reshape %1791 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %1793 = stablehlo.slice %1792 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1794 = stablehlo.reshape %1793 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1795 = stablehlo.slice %1792 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1796 = stablehlo.reshape %1795 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1797 = stablehlo.complex %1794, %1796 : tensor<2x1x32x64xcomplex<f32>>
    %1798 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %1799 = stablehlo.multiply %1797, %1798 : tensor<2x1x32x64xcomplex<f32>>
    %1800 = stablehlo.real %1799 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1801 = stablehlo.reshape %1800 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1802 = stablehlo.imag %1799 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1803 = stablehlo.reshape %1802 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1804 = stablehlo.concatenate %1801, %1803, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %1805 = stablehlo.reshape %1804 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %1806 = "stablehlo.scatter"(%arg280, %1788, %1805) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1807 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1808 = "stablehlo.gather"(%1806, %1807) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1809 = stablehlo.transpose %1808, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %1810 = stablehlo.reshape %1809 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %1811 = stablehlo.dot_general %1783, %1810, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %1812 = stablehlo.reshape %1811 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %1813 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %1814 = stablehlo.divide %1812, %1813 : tensor<2x32x1x2048xf32>
    %1815 = stablehlo.reduce(%1814 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1816 = stablehlo.broadcast_in_dim %1815, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1817 = stablehlo.subtract %1814, %1816 : tensor<2x32x1x2048xf32>
    %1818 = stablehlo.exponential %1817 : tensor<2x32x1x2048xf32>
    %1819 = stablehlo.reduce(%1818 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1820 = stablehlo.broadcast_in_dim %1819, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1821 = stablehlo.divide %1818, %1820 : tensor<2x32x1x2048xf32>
    %1822 = stablehlo.reshape %1821 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %1823 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1824 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1825 = stablehlo.add %arg199, %1824 : tensor<1xi64>
    %1826 = stablehlo.select %1823, %1825, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1827 = stablehlo.reshape %1826 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1828 = stablehlo.reshape %1766 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1829 = stablehlo.transpose %arg104, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1830 = stablehlo.dot %1828, %1829, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1831 = stablehlo.reshape %1830 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %1832 = "stablehlo.scatter"(%arg278, %1827, %1831) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1833 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1834 = "stablehlo.gather"(%1832, %1833) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1835 = stablehlo.transpose %1834, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %1836 = stablehlo.reshape %1835 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %1837 = stablehlo.dot_general %1822, %1836, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %1838 = stablehlo.reshape %1837 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %1839 = stablehlo.transpose %arg103, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1840 = stablehlo.dot %1838, %1839, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1841 = stablehlo.reshape %1840 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1842 = stablehlo.add %1754, %1841 : tensor<2x1x4096xf32>
    %1843 = stablehlo.power %1842, %1 : tensor<2x1x4096xf32>
    %1844 = stablehlo.reduce(%1843 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1845 = stablehlo.multiply %1844, %0 : tensor<2x1xf32>
    %1846 = stablehlo.reshape %1845 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1847 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1848 = stablehlo.add %1846, %1847 : tensor<2x1x1xf32>
    %1849 = stablehlo.rsqrt %1848 : tensor<2x1x1xf32>
    %1850 = stablehlo.reshape %1849 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1851 = stablehlo.broadcast_in_dim %1850, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1852 = stablehlo.multiply %1842, %1851 : tensor<2x1x4096xf32>
    %1853 = stablehlo.broadcast_in_dim %arg102, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1854 = stablehlo.multiply %1852, %1853 : tensor<2x1x4096xf32>
    %1855 = stablehlo.reshape %1854 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1856 = stablehlo.transpose %arg282, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1857 = stablehlo.dot %1855, %1856, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1858 = stablehlo.reshape %1857 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1859 = stablehlo.logistic %1858 : tensor<2x1x11008xf32>
    %1860 = stablehlo.multiply %1858, %1859 : tensor<2x1x11008xf32>
    %1861 = stablehlo.reshape %1854 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1862 = stablehlo.transpose %arg101, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1863 = stablehlo.dot %1861, %1862, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1864 = stablehlo.reshape %1863 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1865 = stablehlo.multiply %1860, %1864 : tensor<2x1x11008xf32>
    %1866 = stablehlo.reshape %1865 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %1867 = stablehlo.transpose %arg100, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1868 = stablehlo.dot %1866, %1867, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %1869 = stablehlo.reshape %1868 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1870 = stablehlo.add %1842, %1869 : tensor<2x1x4096xf32>
    %1871 = stablehlo.power %1870, %1 : tensor<2x1x4096xf32>
    %1872 = stablehlo.reduce(%1871 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1873 = stablehlo.multiply %1872, %0 : tensor<2x1xf32>
    %1874 = stablehlo.reshape %1873 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1875 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1876 = stablehlo.add %1874, %1875 : tensor<2x1x1xf32>
    %1877 = stablehlo.rsqrt %1876 : tensor<2x1x1xf32>
    %1878 = stablehlo.reshape %1877 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1879 = stablehlo.broadcast_in_dim %1878, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1880 = stablehlo.multiply %1870, %1879 : tensor<2x1x4096xf32>
    %1881 = stablehlo.broadcast_in_dim %arg99, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1882 = stablehlo.multiply %1880, %1881 : tensor<2x1x4096xf32>
    %1883 = stablehlo.reshape %1882 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1884 = stablehlo.transpose %arg286, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1885 = stablehlo.dot %1883, %1884, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1886 = stablehlo.reshape %1885 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %1887 = stablehlo.slice %1886 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1888 = stablehlo.reshape %1887 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1889 = stablehlo.slice %1886 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1890 = stablehlo.reshape %1889 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1891 = stablehlo.complex %1888, %1890 : tensor<2x1x32x64xcomplex<f32>>
    %1892 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %1893 = stablehlo.multiply %1891, %1892 : tensor<2x1x32x64xcomplex<f32>>
    %1894 = stablehlo.real %1893 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1895 = stablehlo.reshape %1894 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1896 = stablehlo.imag %1893 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1897 = stablehlo.reshape %1896 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1898 = stablehlo.concatenate %1895, %1897, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %1899 = stablehlo.reshape %1898 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %1900 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1901 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1902 = stablehlo.add %arg199, %1901 : tensor<1xi64>
    %1903 = stablehlo.select %1900, %1902, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1904 = stablehlo.reshape %1903 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1905 = stablehlo.reshape %1882 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1906 = stablehlo.transpose %arg284, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1907 = stablehlo.dot %1905, %1906, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1908 = stablehlo.reshape %1907 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %1909 = stablehlo.slice %1908 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1910 = stablehlo.reshape %1909 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1911 = stablehlo.slice %1908 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %1912 = stablehlo.reshape %1911 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %1913 = stablehlo.complex %1910, %1912 : tensor<2x1x32x64xcomplex<f32>>
    %1914 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %1915 = stablehlo.multiply %1913, %1914 : tensor<2x1x32x64xcomplex<f32>>
    %1916 = stablehlo.real %1915 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1917 = stablehlo.reshape %1916 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1918 = stablehlo.imag %1915 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %1919 = stablehlo.reshape %1918 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %1920 = stablehlo.concatenate %1917, %1919, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %1921 = stablehlo.reshape %1920 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %1922 = "stablehlo.scatter"(%arg285, %1904, %1921) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1923 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1924 = "stablehlo.gather"(%1922, %1923) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1925 = stablehlo.transpose %1924, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %1926 = stablehlo.reshape %1925 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %1927 = stablehlo.dot_general %1899, %1926, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %1928 = stablehlo.reshape %1927 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %1929 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %1930 = stablehlo.divide %1928, %1929 : tensor<2x32x1x2048xf32>
    %1931 = stablehlo.reduce(%1930 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1932 = stablehlo.broadcast_in_dim %1931, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1933 = stablehlo.subtract %1930, %1932 : tensor<2x32x1x2048xf32>
    %1934 = stablehlo.exponential %1933 : tensor<2x32x1x2048xf32>
    %1935 = stablehlo.reduce(%1934 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1936 = stablehlo.broadcast_in_dim %1935, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %1937 = stablehlo.divide %1934, %1936 : tensor<2x32x1x2048xf32>
    %1938 = stablehlo.reshape %1937 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %1939 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %1940 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %1941 = stablehlo.add %arg199, %1940 : tensor<1xi64>
    %1942 = stablehlo.select %1939, %1941, %arg199 : tensor<1xi1>, tensor<1xi64>
    %1943 = stablehlo.reshape %1942 : (tensor<1xi64>) -> tensor<1x1xi64>
    %1944 = stablehlo.reshape %1882 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1945 = stablehlo.transpose %arg98, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1946 = stablehlo.dot %1944, %1945, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1947 = stablehlo.reshape %1946 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %1948 = "stablehlo.scatter"(%arg283, %1943, %1947) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %1949 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %1950 = "stablehlo.gather"(%1948, %1949) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %1951 = stablehlo.transpose %1950, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %1952 = stablehlo.reshape %1951 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %1953 = stablehlo.dot_general %1938, %1952, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %1954 = stablehlo.reshape %1953 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %1955 = stablehlo.transpose %arg97, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %1956 = stablehlo.dot %1954, %1955, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %1957 = stablehlo.reshape %1956 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1958 = stablehlo.add %1870, %1957 : tensor<2x1x4096xf32>
    %1959 = stablehlo.power %1958, %1 : tensor<2x1x4096xf32>
    %1960 = stablehlo.reduce(%1959 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1961 = stablehlo.multiply %1960, %0 : tensor<2x1xf32>
    %1962 = stablehlo.reshape %1961 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1963 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1964 = stablehlo.add %1962, %1963 : tensor<2x1x1xf32>
    %1965 = stablehlo.rsqrt %1964 : tensor<2x1x1xf32>
    %1966 = stablehlo.reshape %1965 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1967 = stablehlo.broadcast_in_dim %1966, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1968 = stablehlo.multiply %1958, %1967 : tensor<2x1x4096xf32>
    %1969 = stablehlo.broadcast_in_dim %arg96, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1970 = stablehlo.multiply %1968, %1969 : tensor<2x1x4096xf32>
    %1971 = stablehlo.reshape %1970 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1972 = stablehlo.transpose %arg287, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1973 = stablehlo.dot %1971, %1972, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1974 = stablehlo.reshape %1973 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1975 = stablehlo.logistic %1974 : tensor<2x1x11008xf32>
    %1976 = stablehlo.multiply %1974, %1975 : tensor<2x1x11008xf32>
    %1977 = stablehlo.reshape %1970 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %1978 = stablehlo.transpose %arg95, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %1979 = stablehlo.dot %1977, %1978, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %1980 = stablehlo.reshape %1979 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %1981 = stablehlo.multiply %1976, %1980 : tensor<2x1x11008xf32>
    %1982 = stablehlo.reshape %1981 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %1983 = stablehlo.transpose %arg94, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %1984 = stablehlo.dot %1982, %1983, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %1985 = stablehlo.reshape %1984 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %1986 = stablehlo.add %1958, %1985 : tensor<2x1x4096xf32>
    %1987 = stablehlo.power %1986, %1 : tensor<2x1x4096xf32>
    %1988 = stablehlo.reduce(%1987 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %1989 = stablehlo.multiply %1988, %0 : tensor<2x1xf32>
    %1990 = stablehlo.reshape %1989 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %1991 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %1992 = stablehlo.add %1990, %1991 : tensor<2x1x1xf32>
    %1993 = stablehlo.rsqrt %1992 : tensor<2x1x1xf32>
    %1994 = stablehlo.reshape %1993 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %1995 = stablehlo.broadcast_in_dim %1994, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %1996 = stablehlo.multiply %1986, %1995 : tensor<2x1x4096xf32>
    %1997 = stablehlo.broadcast_in_dim %arg93, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %1998 = stablehlo.multiply %1996, %1997 : tensor<2x1x4096xf32>
    %1999 = stablehlo.reshape %1998 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2000 = stablehlo.transpose %arg291, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2001 = stablehlo.dot %1999, %2000, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2002 = stablehlo.reshape %2001 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2003 = stablehlo.slice %2002 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2004 = stablehlo.reshape %2003 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2005 = stablehlo.slice %2002 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2006 = stablehlo.reshape %2005 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2007 = stablehlo.complex %2004, %2006 : tensor<2x1x32x64xcomplex<f32>>
    %2008 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2009 = stablehlo.multiply %2007, %2008 : tensor<2x1x32x64xcomplex<f32>>
    %2010 = stablehlo.real %2009 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2011 = stablehlo.reshape %2010 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2012 = stablehlo.imag %2009 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2013 = stablehlo.reshape %2012 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2014 = stablehlo.concatenate %2011, %2013, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2015 = stablehlo.reshape %2014 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %2016 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2017 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2018 = stablehlo.add %arg199, %2017 : tensor<1xi64>
    %2019 = stablehlo.select %2016, %2018, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2020 = stablehlo.reshape %2019 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2021 = stablehlo.reshape %1998 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2022 = stablehlo.transpose %arg289, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2023 = stablehlo.dot %2021, %2022, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2024 = stablehlo.reshape %2023 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2025 = stablehlo.slice %2024 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2026 = stablehlo.reshape %2025 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2027 = stablehlo.slice %2024 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2028 = stablehlo.reshape %2027 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2029 = stablehlo.complex %2026, %2028 : tensor<2x1x32x64xcomplex<f32>>
    %2030 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2031 = stablehlo.multiply %2029, %2030 : tensor<2x1x32x64xcomplex<f32>>
    %2032 = stablehlo.real %2031 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2033 = stablehlo.reshape %2032 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2034 = stablehlo.imag %2031 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2035 = stablehlo.reshape %2034 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2036 = stablehlo.concatenate %2033, %2035, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2037 = stablehlo.reshape %2036 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %2038 = "stablehlo.scatter"(%arg290, %2020, %2037) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2039 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2040 = "stablehlo.gather"(%2038, %2039) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2041 = stablehlo.transpose %2040, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %2042 = stablehlo.reshape %2041 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %2043 = stablehlo.dot_general %2015, %2042, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %2044 = stablehlo.reshape %2043 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %2045 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %2046 = stablehlo.divide %2044, %2045 : tensor<2x32x1x2048xf32>
    %2047 = stablehlo.reduce(%2046 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2048 = stablehlo.broadcast_in_dim %2047, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2049 = stablehlo.subtract %2046, %2048 : tensor<2x32x1x2048xf32>
    %2050 = stablehlo.exponential %2049 : tensor<2x32x1x2048xf32>
    %2051 = stablehlo.reduce(%2050 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2052 = stablehlo.broadcast_in_dim %2051, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2053 = stablehlo.divide %2050, %2052 : tensor<2x32x1x2048xf32>
    %2054 = stablehlo.reshape %2053 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %2055 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2056 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2057 = stablehlo.add %arg199, %2056 : tensor<1xi64>
    %2058 = stablehlo.select %2055, %2057, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2059 = stablehlo.reshape %2058 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2060 = stablehlo.reshape %1998 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2061 = stablehlo.transpose %arg92, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2062 = stablehlo.dot %2060, %2061, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2063 = stablehlo.reshape %2062 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %2064 = "stablehlo.scatter"(%arg288, %2059, %2063) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2065 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2066 = "stablehlo.gather"(%2064, %2065) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2067 = stablehlo.transpose %2066, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %2068 = stablehlo.reshape %2067 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %2069 = stablehlo.dot_general %2054, %2068, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %2070 = stablehlo.reshape %2069 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %2071 = stablehlo.transpose %arg91, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2072 = stablehlo.dot %2070, %2071, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2073 = stablehlo.reshape %2072 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %2074 = stablehlo.add %1986, %2073 : tensor<2x1x4096xf32>
    %2075 = stablehlo.power %2074, %1 : tensor<2x1x4096xf32>
    %2076 = stablehlo.reduce(%2075 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2077 = stablehlo.multiply %2076, %0 : tensor<2x1xf32>
    %2078 = stablehlo.reshape %2077 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %2079 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %2080 = stablehlo.add %2078, %2079 : tensor<2x1x1xf32>
    %2081 = stablehlo.rsqrt %2080 : tensor<2x1x1xf32>
    %2082 = stablehlo.reshape %2081 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %2083 = stablehlo.broadcast_in_dim %2082, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %2084 = stablehlo.multiply %2074, %2083 : tensor<2x1x4096xf32>
    %2085 = stablehlo.broadcast_in_dim %arg90, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %2086 = stablehlo.multiply %2084, %2085 : tensor<2x1x4096xf32>
    %2087 = stablehlo.reshape %2086 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2088 = stablehlo.transpose %arg292, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2089 = stablehlo.dot %2087, %2088, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %2090 = stablehlo.reshape %2089 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %2091 = stablehlo.logistic %2090 : tensor<2x1x11008xf32>
    %2092 = stablehlo.multiply %2090, %2091 : tensor<2x1x11008xf32>
    %2093 = stablehlo.reshape %2086 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2094 = stablehlo.transpose %arg89, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2095 = stablehlo.dot %2093, %2094, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %2096 = stablehlo.reshape %2095 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %2097 = stablehlo.multiply %2092, %2096 : tensor<2x1x11008xf32>
    %2098 = stablehlo.reshape %2097 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %2099 = stablehlo.transpose %arg88, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2100 = stablehlo.dot %2098, %2099, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %2101 = stablehlo.reshape %2100 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %2102 = stablehlo.add %2074, %2101 : tensor<2x1x4096xf32>
    %2103 = stablehlo.power %2102, %1 : tensor<2x1x4096xf32>
    %2104 = stablehlo.reduce(%2103 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2105 = stablehlo.multiply %2104, %0 : tensor<2x1xf32>
    %2106 = stablehlo.reshape %2105 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %2107 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %2108 = stablehlo.add %2106, %2107 : tensor<2x1x1xf32>
    %2109 = stablehlo.rsqrt %2108 : tensor<2x1x1xf32>
    %2110 = stablehlo.reshape %2109 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %2111 = stablehlo.broadcast_in_dim %2110, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %2112 = stablehlo.multiply %2102, %2111 : tensor<2x1x4096xf32>
    %2113 = stablehlo.broadcast_in_dim %arg87, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %2114 = stablehlo.multiply %2112, %2113 : tensor<2x1x4096xf32>
    %2115 = stablehlo.reshape %2114 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2116 = stablehlo.transpose %arg296, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2117 = stablehlo.dot %2115, %2116, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2118 = stablehlo.reshape %2117 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2119 = stablehlo.slice %2118 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2120 = stablehlo.reshape %2119 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2121 = stablehlo.slice %2118 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2122 = stablehlo.reshape %2121 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2123 = stablehlo.complex %2120, %2122 : tensor<2x1x32x64xcomplex<f32>>
    %2124 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2125 = stablehlo.multiply %2123, %2124 : tensor<2x1x32x64xcomplex<f32>>
    %2126 = stablehlo.real %2125 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2127 = stablehlo.reshape %2126 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2128 = stablehlo.imag %2125 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2129 = stablehlo.reshape %2128 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2130 = stablehlo.concatenate %2127, %2129, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2131 = stablehlo.reshape %2130 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %2132 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2133 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2134 = stablehlo.add %arg199, %2133 : tensor<1xi64>
    %2135 = stablehlo.select %2132, %2134, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2136 = stablehlo.reshape %2135 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2137 = stablehlo.reshape %2114 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2138 = stablehlo.transpose %arg294, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2139 = stablehlo.dot %2137, %2138, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2140 = stablehlo.reshape %2139 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2141 = stablehlo.slice %2140 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2142 = stablehlo.reshape %2141 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2143 = stablehlo.slice %2140 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2144 = stablehlo.reshape %2143 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2145 = stablehlo.complex %2142, %2144 : tensor<2x1x32x64xcomplex<f32>>
    %2146 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2147 = stablehlo.multiply %2145, %2146 : tensor<2x1x32x64xcomplex<f32>>
    %2148 = stablehlo.real %2147 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2149 = stablehlo.reshape %2148 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2150 = stablehlo.imag %2147 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2151 = stablehlo.reshape %2150 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2152 = stablehlo.concatenate %2149, %2151, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2153 = stablehlo.reshape %2152 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %2154 = "stablehlo.scatter"(%arg295, %2136, %2153) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2155 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2156 = "stablehlo.gather"(%2154, %2155) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2157 = stablehlo.transpose %2156, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %2158 = stablehlo.reshape %2157 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %2159 = stablehlo.dot_general %2131, %2158, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %2160 = stablehlo.reshape %2159 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %2161 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %2162 = stablehlo.divide %2160, %2161 : tensor<2x32x1x2048xf32>
    %2163 = stablehlo.reduce(%2162 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2164 = stablehlo.broadcast_in_dim %2163, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2165 = stablehlo.subtract %2162, %2164 : tensor<2x32x1x2048xf32>
    %2166 = stablehlo.exponential %2165 : tensor<2x32x1x2048xf32>
    %2167 = stablehlo.reduce(%2166 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2168 = stablehlo.broadcast_in_dim %2167, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2169 = stablehlo.divide %2166, %2168 : tensor<2x32x1x2048xf32>
    %2170 = stablehlo.reshape %2169 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %2171 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2172 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2173 = stablehlo.add %arg199, %2172 : tensor<1xi64>
    %2174 = stablehlo.select %2171, %2173, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2175 = stablehlo.reshape %2174 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2176 = stablehlo.reshape %2114 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2177 = stablehlo.transpose %arg86, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2178 = stablehlo.dot %2176, %2177, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2179 = stablehlo.reshape %2178 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %2180 = "stablehlo.scatter"(%arg293, %2175, %2179) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2181 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2182 = "stablehlo.gather"(%2180, %2181) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2183 = stablehlo.transpose %2182, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %2184 = stablehlo.reshape %2183 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %2185 = stablehlo.dot_general %2170, %2184, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %2186 = stablehlo.reshape %2185 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %2187 = stablehlo.transpose %arg85, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2188 = stablehlo.dot %2186, %2187, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2189 = stablehlo.reshape %2188 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %2190 = stablehlo.add %2102, %2189 : tensor<2x1x4096xf32>
    %2191 = stablehlo.power %2190, %1 : tensor<2x1x4096xf32>
    %2192 = stablehlo.reduce(%2191 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2193 = stablehlo.multiply %2192, %0 : tensor<2x1xf32>
    %2194 = stablehlo.reshape %2193 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %2195 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %2196 = stablehlo.add %2194, %2195 : tensor<2x1x1xf32>
    %2197 = stablehlo.rsqrt %2196 : tensor<2x1x1xf32>
    %2198 = stablehlo.reshape %2197 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %2199 = stablehlo.broadcast_in_dim %2198, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %2200 = stablehlo.multiply %2190, %2199 : tensor<2x1x4096xf32>
    %2201 = stablehlo.broadcast_in_dim %arg84, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %2202 = stablehlo.multiply %2200, %2201 : tensor<2x1x4096xf32>
    %2203 = stablehlo.reshape %2202 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2204 = stablehlo.transpose %arg297, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2205 = stablehlo.dot %2203, %2204, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %2206 = stablehlo.reshape %2205 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %2207 = stablehlo.logistic %2206 : tensor<2x1x11008xf32>
    %2208 = stablehlo.multiply %2206, %2207 : tensor<2x1x11008xf32>
    %2209 = stablehlo.reshape %2202 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2210 = stablehlo.transpose %arg83, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2211 = stablehlo.dot %2209, %2210, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %2212 = stablehlo.reshape %2211 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %2213 = stablehlo.multiply %2208, %2212 : tensor<2x1x11008xf32>
    %2214 = stablehlo.reshape %2213 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %2215 = stablehlo.transpose %arg82, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2216 = stablehlo.dot %2214, %2215, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %2217 = stablehlo.reshape %2216 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %2218 = stablehlo.add %2190, %2217 : tensor<2x1x4096xf32>
    %2219 = stablehlo.power %2218, %1 : tensor<2x1x4096xf32>
    %2220 = stablehlo.reduce(%2219 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2221 = stablehlo.multiply %2220, %0 : tensor<2x1xf32>
    %2222 = stablehlo.reshape %2221 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %2223 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %2224 = stablehlo.add %2222, %2223 : tensor<2x1x1xf32>
    %2225 = stablehlo.rsqrt %2224 : tensor<2x1x1xf32>
    %2226 = stablehlo.reshape %2225 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %2227 = stablehlo.broadcast_in_dim %2226, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %2228 = stablehlo.multiply %2218, %2227 : tensor<2x1x4096xf32>
    %2229 = stablehlo.broadcast_in_dim %arg81, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %2230 = stablehlo.multiply %2228, %2229 : tensor<2x1x4096xf32>
    %2231 = stablehlo.reshape %2230 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2232 = stablehlo.transpose %arg301, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2233 = stablehlo.dot %2231, %2232, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2234 = stablehlo.reshape %2233 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2235 = stablehlo.slice %2234 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2236 = stablehlo.reshape %2235 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2237 = stablehlo.slice %2234 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2238 = stablehlo.reshape %2237 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2239 = stablehlo.complex %2236, %2238 : tensor<2x1x32x64xcomplex<f32>>
    %2240 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2241 = stablehlo.multiply %2239, %2240 : tensor<2x1x32x64xcomplex<f32>>
    %2242 = stablehlo.real %2241 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2243 = stablehlo.reshape %2242 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2244 = stablehlo.imag %2241 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2245 = stablehlo.reshape %2244 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2246 = stablehlo.concatenate %2243, %2245, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2247 = stablehlo.reshape %2246 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %2248 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2249 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2250 = stablehlo.add %arg199, %2249 : tensor<1xi64>
    %2251 = stablehlo.select %2248, %2250, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2252 = stablehlo.reshape %2251 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2253 = stablehlo.reshape %2230 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2254 = stablehlo.transpose %arg299, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2255 = stablehlo.dot %2253, %2254, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2256 = stablehlo.reshape %2255 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2257 = stablehlo.slice %2256 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2258 = stablehlo.reshape %2257 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2259 = stablehlo.slice %2256 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2260 = stablehlo.reshape %2259 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2261 = stablehlo.complex %2258, %2260 : tensor<2x1x32x64xcomplex<f32>>
    %2262 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2263 = stablehlo.multiply %2261, %2262 : tensor<2x1x32x64xcomplex<f32>>
    %2264 = stablehlo.real %2263 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2265 = stablehlo.reshape %2264 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2266 = stablehlo.imag %2263 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2267 = stablehlo.reshape %2266 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2268 = stablehlo.concatenate %2265, %2267, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2269 = stablehlo.reshape %2268 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %2270 = "stablehlo.scatter"(%arg300, %2252, %2269) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2271 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2272 = "stablehlo.gather"(%2270, %2271) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2273 = stablehlo.transpose %2272, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %2274 = stablehlo.reshape %2273 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %2275 = stablehlo.dot_general %2247, %2274, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %2276 = stablehlo.reshape %2275 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %2277 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %2278 = stablehlo.divide %2276, %2277 : tensor<2x32x1x2048xf32>
    %2279 = stablehlo.reduce(%2278 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2280 = stablehlo.broadcast_in_dim %2279, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2281 = stablehlo.subtract %2278, %2280 : tensor<2x32x1x2048xf32>
    %2282 = stablehlo.exponential %2281 : tensor<2x32x1x2048xf32>
    %2283 = stablehlo.reduce(%2282 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2284 = stablehlo.broadcast_in_dim %2283, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2285 = stablehlo.divide %2282, %2284 : tensor<2x32x1x2048xf32>
    %2286 = stablehlo.reshape %2285 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %2287 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2288 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2289 = stablehlo.add %arg199, %2288 : tensor<1xi64>
    %2290 = stablehlo.select %2287, %2289, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2291 = stablehlo.reshape %2290 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2292 = stablehlo.reshape %2230 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2293 = stablehlo.transpose %arg80, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2294 = stablehlo.dot %2292, %2293, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2295 = stablehlo.reshape %2294 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %2296 = "stablehlo.scatter"(%arg298, %2291, %2295) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2297 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2298 = "stablehlo.gather"(%2296, %2297) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2299 = stablehlo.transpose %2298, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %2300 = stablehlo.reshape %2299 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %2301 = stablehlo.dot_general %2286, %2300, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %2302 = stablehlo.reshape %2301 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %2303 = stablehlo.transpose %arg79, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2304 = stablehlo.dot %2302, %2303, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2305 = stablehlo.reshape %2304 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %2306 = stablehlo.add %2218, %2305 : tensor<2x1x4096xf32>
    %2307 = stablehlo.power %2306, %1 : tensor<2x1x4096xf32>
    %2308 = stablehlo.reduce(%2307 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2309 = stablehlo.multiply %2308, %0 : tensor<2x1xf32>
    %2310 = stablehlo.reshape %2309 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %2311 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %2312 = stablehlo.add %2310, %2311 : tensor<2x1x1xf32>
    %2313 = stablehlo.rsqrt %2312 : tensor<2x1x1xf32>
    %2314 = stablehlo.reshape %2313 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %2315 = stablehlo.broadcast_in_dim %2314, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %2316 = stablehlo.multiply %2306, %2315 : tensor<2x1x4096xf32>
    %2317 = stablehlo.broadcast_in_dim %arg78, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %2318 = stablehlo.multiply %2316, %2317 : tensor<2x1x4096xf32>
    %2319 = stablehlo.reshape %2318 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2320 = stablehlo.transpose %arg302, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2321 = stablehlo.dot %2319, %2320, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %2322 = stablehlo.reshape %2321 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %2323 = stablehlo.logistic %2322 : tensor<2x1x11008xf32>
    %2324 = stablehlo.multiply %2322, %2323 : tensor<2x1x11008xf32>
    %2325 = stablehlo.reshape %2318 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2326 = stablehlo.transpose %arg77, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2327 = stablehlo.dot %2325, %2326, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %2328 = stablehlo.reshape %2327 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %2329 = stablehlo.multiply %2324, %2328 : tensor<2x1x11008xf32>
    %2330 = stablehlo.reshape %2329 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %2331 = stablehlo.transpose %arg76, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2332 = stablehlo.dot %2330, %2331, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %2333 = stablehlo.reshape %2332 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %2334 = stablehlo.add %2306, %2333 : tensor<2x1x4096xf32>
    %2335 = stablehlo.power %2334, %1 : tensor<2x1x4096xf32>
    %2336 = stablehlo.reduce(%2335 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2337 = stablehlo.multiply %2336, %0 : tensor<2x1xf32>
    %2338 = stablehlo.reshape %2337 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %2339 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %2340 = stablehlo.add %2338, %2339 : tensor<2x1x1xf32>
    %2341 = stablehlo.rsqrt %2340 : tensor<2x1x1xf32>
    %2342 = stablehlo.reshape %2341 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %2343 = stablehlo.broadcast_in_dim %2342, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %2344 = stablehlo.multiply %2334, %2343 : tensor<2x1x4096xf32>
    %2345 = stablehlo.broadcast_in_dim %arg75, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %2346 = stablehlo.multiply %2344, %2345 : tensor<2x1x4096xf32>
    %2347 = stablehlo.reshape %2346 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2348 = stablehlo.transpose %arg306, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2349 = stablehlo.dot %2347, %2348, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2350 = stablehlo.reshape %2349 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2351 = stablehlo.slice %2350 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2352 = stablehlo.reshape %2351 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2353 = stablehlo.slice %2350 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2354 = stablehlo.reshape %2353 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2355 = stablehlo.complex %2352, %2354 : tensor<2x1x32x64xcomplex<f32>>
    %2356 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2357 = stablehlo.multiply %2355, %2356 : tensor<2x1x32x64xcomplex<f32>>
    %2358 = stablehlo.real %2357 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2359 = stablehlo.reshape %2358 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2360 = stablehlo.imag %2357 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2361 = stablehlo.reshape %2360 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2362 = stablehlo.concatenate %2359, %2361, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2363 = stablehlo.reshape %2362 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %2364 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2365 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2366 = stablehlo.add %arg199, %2365 : tensor<1xi64>
    %2367 = stablehlo.select %2364, %2366, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2368 = stablehlo.reshape %2367 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2369 = stablehlo.reshape %2346 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2370 = stablehlo.transpose %arg304, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2371 = stablehlo.dot %2369, %2370, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2372 = stablehlo.reshape %2371 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2373 = stablehlo.slice %2372 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2374 = stablehlo.reshape %2373 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2375 = stablehlo.slice %2372 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2376 = stablehlo.reshape %2375 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2377 = stablehlo.complex %2374, %2376 : tensor<2x1x32x64xcomplex<f32>>
    %2378 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2379 = stablehlo.multiply %2377, %2378 : tensor<2x1x32x64xcomplex<f32>>
    %2380 = stablehlo.real %2379 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2381 = stablehlo.reshape %2380 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2382 = stablehlo.imag %2379 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2383 = stablehlo.reshape %2382 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2384 = stablehlo.concatenate %2381, %2383, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2385 = stablehlo.reshape %2384 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %2386 = "stablehlo.scatter"(%arg305, %2368, %2385) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2387 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2388 = "stablehlo.gather"(%2386, %2387) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2389 = stablehlo.transpose %2388, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %2390 = stablehlo.reshape %2389 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %2391 = stablehlo.dot_general %2363, %2390, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %2392 = stablehlo.reshape %2391 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %2393 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %2394 = stablehlo.divide %2392, %2393 : tensor<2x32x1x2048xf32>
    %2395 = stablehlo.reduce(%2394 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2396 = stablehlo.broadcast_in_dim %2395, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2397 = stablehlo.subtract %2394, %2396 : tensor<2x32x1x2048xf32>
    %2398 = stablehlo.exponential %2397 : tensor<2x32x1x2048xf32>
    %2399 = stablehlo.reduce(%2398 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2400 = stablehlo.broadcast_in_dim %2399, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2401 = stablehlo.divide %2398, %2400 : tensor<2x32x1x2048xf32>
    %2402 = stablehlo.reshape %2401 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %2403 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2404 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2405 = stablehlo.add %arg199, %2404 : tensor<1xi64>
    %2406 = stablehlo.select %2403, %2405, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2407 = stablehlo.reshape %2406 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2408 = stablehlo.reshape %2346 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2409 = stablehlo.transpose %arg74, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2410 = stablehlo.dot %2408, %2409, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2411 = stablehlo.reshape %2410 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %2412 = "stablehlo.scatter"(%arg303, %2407, %2411) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2413 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2414 = "stablehlo.gather"(%2412, %2413) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2415 = stablehlo.transpose %2414, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %2416 = stablehlo.reshape %2415 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %2417 = stablehlo.dot_general %2402, %2416, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %2418 = stablehlo.reshape %2417 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %2419 = stablehlo.transpose %arg73, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2420 = stablehlo.dot %2418, %2419, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2421 = stablehlo.reshape %2420 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %2422 = stablehlo.add %2334, %2421 : tensor<2x1x4096xf32>
    %2423 = stablehlo.power %2422, %1 : tensor<2x1x4096xf32>
    %2424 = stablehlo.reduce(%2423 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2425 = stablehlo.multiply %2424, %0 : tensor<2x1xf32>
    %2426 = stablehlo.reshape %2425 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %2427 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %2428 = stablehlo.add %2426, %2427 : tensor<2x1x1xf32>
    %2429 = stablehlo.rsqrt %2428 : tensor<2x1x1xf32>
    %2430 = stablehlo.reshape %2429 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %2431 = stablehlo.broadcast_in_dim %2430, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %2432 = stablehlo.multiply %2422, %2431 : tensor<2x1x4096xf32>
    %2433 = stablehlo.broadcast_in_dim %arg72, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %2434 = stablehlo.multiply %2432, %2433 : tensor<2x1x4096xf32>
    %2435 = stablehlo.reshape %2434 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2436 = stablehlo.transpose %arg307, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2437 = stablehlo.dot %2435, %2436, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %2438 = stablehlo.reshape %2437 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %2439 = stablehlo.logistic %2438 : tensor<2x1x11008xf32>
    %2440 = stablehlo.multiply %2438, %2439 : tensor<2x1x11008xf32>
    %2441 = stablehlo.reshape %2434 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2442 = stablehlo.transpose %arg71, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2443 = stablehlo.dot %2441, %2442, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %2444 = stablehlo.reshape %2443 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %2445 = stablehlo.multiply %2440, %2444 : tensor<2x1x11008xf32>
    %2446 = stablehlo.reshape %2445 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %2447 = stablehlo.transpose %arg70, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2448 = stablehlo.dot %2446, %2447, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %2449 = stablehlo.reshape %2448 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %2450 = stablehlo.add %2422, %2449 : tensor<2x1x4096xf32>
    %2451 = stablehlo.power %2450, %1 : tensor<2x1x4096xf32>
    %2452 = stablehlo.reduce(%2451 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2453 = stablehlo.multiply %2452, %0 : tensor<2x1xf32>
    %2454 = stablehlo.reshape %2453 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %2455 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %2456 = stablehlo.add %2454, %2455 : tensor<2x1x1xf32>
    %2457 = stablehlo.rsqrt %2456 : tensor<2x1x1xf32>
    %2458 = stablehlo.reshape %2457 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %2459 = stablehlo.broadcast_in_dim %2458, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %2460 = stablehlo.multiply %2450, %2459 : tensor<2x1x4096xf32>
    %2461 = stablehlo.broadcast_in_dim %arg69, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %2462 = stablehlo.multiply %2460, %2461 : tensor<2x1x4096xf32>
    %2463 = stablehlo.reshape %2462 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2464 = stablehlo.transpose %arg311, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2465 = stablehlo.dot %2463, %2464, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2466 = stablehlo.reshape %2465 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2467 = stablehlo.slice %2466 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2468 = stablehlo.reshape %2467 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2469 = stablehlo.slice %2466 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2470 = stablehlo.reshape %2469 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2471 = stablehlo.complex %2468, %2470 : tensor<2x1x32x64xcomplex<f32>>
    %2472 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2473 = stablehlo.multiply %2471, %2472 : tensor<2x1x32x64xcomplex<f32>>
    %2474 = stablehlo.real %2473 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2475 = stablehlo.reshape %2474 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2476 = stablehlo.imag %2473 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2477 = stablehlo.reshape %2476 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2478 = stablehlo.concatenate %2475, %2477, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2479 = stablehlo.reshape %2478 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %2480 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2481 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2482 = stablehlo.add %arg199, %2481 : tensor<1xi64>
    %2483 = stablehlo.select %2480, %2482, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2484 = stablehlo.reshape %2483 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2485 = stablehlo.reshape %2462 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2486 = stablehlo.transpose %arg309, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2487 = stablehlo.dot %2485, %2486, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2488 = stablehlo.reshape %2487 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2489 = stablehlo.slice %2488 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2490 = stablehlo.reshape %2489 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2491 = stablehlo.slice %2488 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2492 = stablehlo.reshape %2491 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2493 = stablehlo.complex %2490, %2492 : tensor<2x1x32x64xcomplex<f32>>
    %2494 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2495 = stablehlo.multiply %2493, %2494 : tensor<2x1x32x64xcomplex<f32>>
    %2496 = stablehlo.real %2495 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2497 = stablehlo.reshape %2496 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2498 = stablehlo.imag %2495 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2499 = stablehlo.reshape %2498 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2500 = stablehlo.concatenate %2497, %2499, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2501 = stablehlo.reshape %2500 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %2502 = "stablehlo.scatter"(%arg310, %2484, %2501) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2503 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2504 = "stablehlo.gather"(%2502, %2503) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2505 = stablehlo.transpose %2504, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %2506 = stablehlo.reshape %2505 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %2507 = stablehlo.dot_general %2479, %2506, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %2508 = stablehlo.reshape %2507 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %2509 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %2510 = stablehlo.divide %2508, %2509 : tensor<2x32x1x2048xf32>
    %2511 = stablehlo.reduce(%2510 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2512 = stablehlo.broadcast_in_dim %2511, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2513 = stablehlo.subtract %2510, %2512 : tensor<2x32x1x2048xf32>
    %2514 = stablehlo.exponential %2513 : tensor<2x32x1x2048xf32>
    %2515 = stablehlo.reduce(%2514 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2516 = stablehlo.broadcast_in_dim %2515, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2517 = stablehlo.divide %2514, %2516 : tensor<2x32x1x2048xf32>
    %2518 = stablehlo.reshape %2517 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %2519 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2520 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2521 = stablehlo.add %arg199, %2520 : tensor<1xi64>
    %2522 = stablehlo.select %2519, %2521, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2523 = stablehlo.reshape %2522 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2524 = stablehlo.reshape %2462 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2525 = stablehlo.transpose %arg68, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2526 = stablehlo.dot %2524, %2525, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2527 = stablehlo.reshape %2526 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %2528 = "stablehlo.scatter"(%arg308, %2523, %2527) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2529 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2530 = "stablehlo.gather"(%2528, %2529) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2531 = stablehlo.transpose %2530, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %2532 = stablehlo.reshape %2531 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %2533 = stablehlo.dot_general %2518, %2532, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %2534 = stablehlo.reshape %2533 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %2535 = stablehlo.transpose %arg67, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2536 = stablehlo.dot %2534, %2535, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2537 = stablehlo.reshape %2536 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %2538 = stablehlo.add %2450, %2537 : tensor<2x1x4096xf32>
    %2539 = stablehlo.power %2538, %1 : tensor<2x1x4096xf32>
    %2540 = stablehlo.reduce(%2539 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2541 = stablehlo.multiply %2540, %0 : tensor<2x1xf32>
    %2542 = stablehlo.reshape %2541 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %2543 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %2544 = stablehlo.add %2542, %2543 : tensor<2x1x1xf32>
    %2545 = stablehlo.rsqrt %2544 : tensor<2x1x1xf32>
    %2546 = stablehlo.reshape %2545 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %2547 = stablehlo.broadcast_in_dim %2546, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %2548 = stablehlo.multiply %2538, %2547 : tensor<2x1x4096xf32>
    %2549 = stablehlo.broadcast_in_dim %arg66, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %2550 = stablehlo.multiply %2548, %2549 : tensor<2x1x4096xf32>
    %2551 = stablehlo.reshape %2550 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2552 = stablehlo.transpose %arg312, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2553 = stablehlo.dot %2551, %2552, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %2554 = stablehlo.reshape %2553 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %2555 = stablehlo.logistic %2554 : tensor<2x1x11008xf32>
    %2556 = stablehlo.multiply %2554, %2555 : tensor<2x1x11008xf32>
    %2557 = stablehlo.reshape %2550 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2558 = stablehlo.transpose %arg65, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2559 = stablehlo.dot %2557, %2558, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %2560 = stablehlo.reshape %2559 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %2561 = stablehlo.multiply %2556, %2560 : tensor<2x1x11008xf32>
    %2562 = stablehlo.reshape %2561 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %2563 = stablehlo.transpose %arg64, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2564 = stablehlo.dot %2562, %2563, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %2565 = stablehlo.reshape %2564 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %2566 = stablehlo.add %2538, %2565 : tensor<2x1x4096xf32>
    %2567 = stablehlo.power %2566, %1 : tensor<2x1x4096xf32>
    %2568 = stablehlo.reduce(%2567 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2569 = stablehlo.multiply %2568, %0 : tensor<2x1xf32>
    %2570 = stablehlo.reshape %2569 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %2571 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %2572 = stablehlo.add %2570, %2571 : tensor<2x1x1xf32>
    %2573 = stablehlo.rsqrt %2572 : tensor<2x1x1xf32>
    %2574 = stablehlo.reshape %2573 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %2575 = stablehlo.broadcast_in_dim %2574, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %2576 = stablehlo.multiply %2566, %2575 : tensor<2x1x4096xf32>
    %2577 = stablehlo.broadcast_in_dim %arg63, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %2578 = stablehlo.multiply %2576, %2577 : tensor<2x1x4096xf32>
    %2579 = stablehlo.reshape %2578 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2580 = stablehlo.transpose %arg316, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2581 = stablehlo.dot %2579, %2580, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2582 = stablehlo.reshape %2581 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2583 = stablehlo.slice %2582 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2584 = stablehlo.reshape %2583 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2585 = stablehlo.slice %2582 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2586 = stablehlo.reshape %2585 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2587 = stablehlo.complex %2584, %2586 : tensor<2x1x32x64xcomplex<f32>>
    %2588 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2589 = stablehlo.multiply %2587, %2588 : tensor<2x1x32x64xcomplex<f32>>
    %2590 = stablehlo.real %2589 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2591 = stablehlo.reshape %2590 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2592 = stablehlo.imag %2589 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2593 = stablehlo.reshape %2592 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2594 = stablehlo.concatenate %2591, %2593, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2595 = stablehlo.reshape %2594 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %2596 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2597 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2598 = stablehlo.add %arg199, %2597 : tensor<1xi64>
    %2599 = stablehlo.select %2596, %2598, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2600 = stablehlo.reshape %2599 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2601 = stablehlo.reshape %2578 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2602 = stablehlo.transpose %arg314, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2603 = stablehlo.dot %2601, %2602, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2604 = stablehlo.reshape %2603 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2605 = stablehlo.slice %2604 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2606 = stablehlo.reshape %2605 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2607 = stablehlo.slice %2604 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2608 = stablehlo.reshape %2607 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2609 = stablehlo.complex %2606, %2608 : tensor<2x1x32x64xcomplex<f32>>
    %2610 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2611 = stablehlo.multiply %2609, %2610 : tensor<2x1x32x64xcomplex<f32>>
    %2612 = stablehlo.real %2611 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2613 = stablehlo.reshape %2612 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2614 = stablehlo.imag %2611 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2615 = stablehlo.reshape %2614 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2616 = stablehlo.concatenate %2613, %2615, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2617 = stablehlo.reshape %2616 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %2618 = "stablehlo.scatter"(%arg315, %2600, %2617) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2619 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2620 = "stablehlo.gather"(%2618, %2619) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2621 = stablehlo.transpose %2620, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %2622 = stablehlo.reshape %2621 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %2623 = stablehlo.dot_general %2595, %2622, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %2624 = stablehlo.reshape %2623 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %2625 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %2626 = stablehlo.divide %2624, %2625 : tensor<2x32x1x2048xf32>
    %2627 = stablehlo.reduce(%2626 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2628 = stablehlo.broadcast_in_dim %2627, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2629 = stablehlo.subtract %2626, %2628 : tensor<2x32x1x2048xf32>
    %2630 = stablehlo.exponential %2629 : tensor<2x32x1x2048xf32>
    %2631 = stablehlo.reduce(%2630 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2632 = stablehlo.broadcast_in_dim %2631, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2633 = stablehlo.divide %2630, %2632 : tensor<2x32x1x2048xf32>
    %2634 = stablehlo.reshape %2633 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %2635 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2636 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2637 = stablehlo.add %arg199, %2636 : tensor<1xi64>
    %2638 = stablehlo.select %2635, %2637, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2639 = stablehlo.reshape %2638 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2640 = stablehlo.reshape %2578 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2641 = stablehlo.transpose %arg62, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2642 = stablehlo.dot %2640, %2641, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2643 = stablehlo.reshape %2642 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %2644 = "stablehlo.scatter"(%arg313, %2639, %2643) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2645 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2646 = "stablehlo.gather"(%2644, %2645) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2647 = stablehlo.transpose %2646, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %2648 = stablehlo.reshape %2647 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %2649 = stablehlo.dot_general %2634, %2648, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %2650 = stablehlo.reshape %2649 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %2651 = stablehlo.transpose %arg61, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2652 = stablehlo.dot %2650, %2651, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2653 = stablehlo.reshape %2652 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %2654 = stablehlo.add %2566, %2653 : tensor<2x1x4096xf32>
    %2655 = stablehlo.power %2654, %1 : tensor<2x1x4096xf32>
    %2656 = stablehlo.reduce(%2655 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2657 = stablehlo.multiply %2656, %0 : tensor<2x1xf32>
    %2658 = stablehlo.reshape %2657 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %2659 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %2660 = stablehlo.add %2658, %2659 : tensor<2x1x1xf32>
    %2661 = stablehlo.rsqrt %2660 : tensor<2x1x1xf32>
    %2662 = stablehlo.reshape %2661 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %2663 = stablehlo.broadcast_in_dim %2662, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %2664 = stablehlo.multiply %2654, %2663 : tensor<2x1x4096xf32>
    %2665 = stablehlo.broadcast_in_dim %arg60, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %2666 = stablehlo.multiply %2664, %2665 : tensor<2x1x4096xf32>
    %2667 = stablehlo.reshape %2666 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2668 = stablehlo.transpose %arg317, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2669 = stablehlo.dot %2667, %2668, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %2670 = stablehlo.reshape %2669 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %2671 = stablehlo.logistic %2670 : tensor<2x1x11008xf32>
    %2672 = stablehlo.multiply %2670, %2671 : tensor<2x1x11008xf32>
    %2673 = stablehlo.reshape %2666 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2674 = stablehlo.transpose %arg59, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2675 = stablehlo.dot %2673, %2674, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %2676 = stablehlo.reshape %2675 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %2677 = stablehlo.multiply %2672, %2676 : tensor<2x1x11008xf32>
    %2678 = stablehlo.reshape %2677 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %2679 = stablehlo.transpose %arg58, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2680 = stablehlo.dot %2678, %2679, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %2681 = stablehlo.reshape %2680 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %2682 = stablehlo.add %2654, %2681 : tensor<2x1x4096xf32>
    %2683 = stablehlo.power %2682, %1 : tensor<2x1x4096xf32>
    %2684 = stablehlo.reduce(%2683 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2685 = stablehlo.multiply %2684, %0 : tensor<2x1xf32>
    %2686 = stablehlo.reshape %2685 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %2687 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %2688 = stablehlo.add %2686, %2687 : tensor<2x1x1xf32>
    %2689 = stablehlo.rsqrt %2688 : tensor<2x1x1xf32>
    %2690 = stablehlo.reshape %2689 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %2691 = stablehlo.broadcast_in_dim %2690, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %2692 = stablehlo.multiply %2682, %2691 : tensor<2x1x4096xf32>
    %2693 = stablehlo.broadcast_in_dim %arg57, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %2694 = stablehlo.multiply %2692, %2693 : tensor<2x1x4096xf32>
    %2695 = stablehlo.reshape %2694 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2696 = stablehlo.transpose %arg321, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2697 = stablehlo.dot %2695, %2696, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2698 = stablehlo.reshape %2697 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2699 = stablehlo.slice %2698 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2700 = stablehlo.reshape %2699 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2701 = stablehlo.slice %2698 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2702 = stablehlo.reshape %2701 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2703 = stablehlo.complex %2700, %2702 : tensor<2x1x32x64xcomplex<f32>>
    %2704 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2705 = stablehlo.multiply %2703, %2704 : tensor<2x1x32x64xcomplex<f32>>
    %2706 = stablehlo.real %2705 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2707 = stablehlo.reshape %2706 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2708 = stablehlo.imag %2705 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2709 = stablehlo.reshape %2708 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2710 = stablehlo.concatenate %2707, %2709, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2711 = stablehlo.reshape %2710 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %2712 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2713 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2714 = stablehlo.add %arg199, %2713 : tensor<1xi64>
    %2715 = stablehlo.select %2712, %2714, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2716 = stablehlo.reshape %2715 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2717 = stablehlo.reshape %2694 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2718 = stablehlo.transpose %arg319, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2719 = stablehlo.dot %2717, %2718, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2720 = stablehlo.reshape %2719 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2721 = stablehlo.slice %2720 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2722 = stablehlo.reshape %2721 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2723 = stablehlo.slice %2720 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2724 = stablehlo.reshape %2723 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2725 = stablehlo.complex %2722, %2724 : tensor<2x1x32x64xcomplex<f32>>
    %2726 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2727 = stablehlo.multiply %2725, %2726 : tensor<2x1x32x64xcomplex<f32>>
    %2728 = stablehlo.real %2727 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2729 = stablehlo.reshape %2728 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2730 = stablehlo.imag %2727 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2731 = stablehlo.reshape %2730 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2732 = stablehlo.concatenate %2729, %2731, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2733 = stablehlo.reshape %2732 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %2734 = "stablehlo.scatter"(%arg320, %2716, %2733) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2735 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2736 = "stablehlo.gather"(%2734, %2735) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2737 = stablehlo.transpose %2736, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %2738 = stablehlo.reshape %2737 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %2739 = stablehlo.dot_general %2711, %2738, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %2740 = stablehlo.reshape %2739 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %2741 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %2742 = stablehlo.divide %2740, %2741 : tensor<2x32x1x2048xf32>
    %2743 = stablehlo.reduce(%2742 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2744 = stablehlo.broadcast_in_dim %2743, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2745 = stablehlo.subtract %2742, %2744 : tensor<2x32x1x2048xf32>
    %2746 = stablehlo.exponential %2745 : tensor<2x32x1x2048xf32>
    %2747 = stablehlo.reduce(%2746 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2748 = stablehlo.broadcast_in_dim %2747, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2749 = stablehlo.divide %2746, %2748 : tensor<2x32x1x2048xf32>
    %2750 = stablehlo.reshape %2749 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %2751 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2752 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2753 = stablehlo.add %arg199, %2752 : tensor<1xi64>
    %2754 = stablehlo.select %2751, %2753, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2755 = stablehlo.reshape %2754 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2756 = stablehlo.reshape %2694 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2757 = stablehlo.transpose %arg56, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2758 = stablehlo.dot %2756, %2757, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2759 = stablehlo.reshape %2758 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %2760 = "stablehlo.scatter"(%arg318, %2755, %2759) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2761 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2762 = "stablehlo.gather"(%2760, %2761) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2763 = stablehlo.transpose %2762, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %2764 = stablehlo.reshape %2763 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %2765 = stablehlo.dot_general %2750, %2764, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %2766 = stablehlo.reshape %2765 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %2767 = stablehlo.transpose %arg55, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2768 = stablehlo.dot %2766, %2767, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2769 = stablehlo.reshape %2768 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %2770 = stablehlo.add %2682, %2769 : tensor<2x1x4096xf32>
    %2771 = stablehlo.power %2770, %1 : tensor<2x1x4096xf32>
    %2772 = stablehlo.reduce(%2771 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2773 = stablehlo.multiply %2772, %0 : tensor<2x1xf32>
    %2774 = stablehlo.reshape %2773 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %2775 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %2776 = stablehlo.add %2774, %2775 : tensor<2x1x1xf32>
    %2777 = stablehlo.rsqrt %2776 : tensor<2x1x1xf32>
    %2778 = stablehlo.reshape %2777 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %2779 = stablehlo.broadcast_in_dim %2778, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %2780 = stablehlo.multiply %2770, %2779 : tensor<2x1x4096xf32>
    %2781 = stablehlo.broadcast_in_dim %arg54, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %2782 = stablehlo.multiply %2780, %2781 : tensor<2x1x4096xf32>
    %2783 = stablehlo.reshape %2782 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2784 = stablehlo.transpose %arg322, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2785 = stablehlo.dot %2783, %2784, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %2786 = stablehlo.reshape %2785 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %2787 = stablehlo.logistic %2786 : tensor<2x1x11008xf32>
    %2788 = stablehlo.multiply %2786, %2787 : tensor<2x1x11008xf32>
    %2789 = stablehlo.reshape %2782 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2790 = stablehlo.transpose %arg53, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2791 = stablehlo.dot %2789, %2790, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %2792 = stablehlo.reshape %2791 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %2793 = stablehlo.multiply %2788, %2792 : tensor<2x1x11008xf32>
    %2794 = stablehlo.reshape %2793 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %2795 = stablehlo.transpose %arg52, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2796 = stablehlo.dot %2794, %2795, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %2797 = stablehlo.reshape %2796 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %2798 = stablehlo.add %2770, %2797 : tensor<2x1x4096xf32>
    %2799 = stablehlo.power %2798, %1 : tensor<2x1x4096xf32>
    %2800 = stablehlo.reduce(%2799 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2801 = stablehlo.multiply %2800, %0 : tensor<2x1xf32>
    %2802 = stablehlo.reshape %2801 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %2803 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %2804 = stablehlo.add %2802, %2803 : tensor<2x1x1xf32>
    %2805 = stablehlo.rsqrt %2804 : tensor<2x1x1xf32>
    %2806 = stablehlo.reshape %2805 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %2807 = stablehlo.broadcast_in_dim %2806, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %2808 = stablehlo.multiply %2798, %2807 : tensor<2x1x4096xf32>
    %2809 = stablehlo.broadcast_in_dim %arg51, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %2810 = stablehlo.multiply %2808, %2809 : tensor<2x1x4096xf32>
    %2811 = stablehlo.reshape %2810 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2812 = stablehlo.transpose %arg326, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2813 = stablehlo.dot %2811, %2812, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2814 = stablehlo.reshape %2813 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2815 = stablehlo.slice %2814 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2816 = stablehlo.reshape %2815 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2817 = stablehlo.slice %2814 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2818 = stablehlo.reshape %2817 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2819 = stablehlo.complex %2816, %2818 : tensor<2x1x32x64xcomplex<f32>>
    %2820 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2821 = stablehlo.multiply %2819, %2820 : tensor<2x1x32x64xcomplex<f32>>
    %2822 = stablehlo.real %2821 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2823 = stablehlo.reshape %2822 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2824 = stablehlo.imag %2821 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2825 = stablehlo.reshape %2824 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2826 = stablehlo.concatenate %2823, %2825, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2827 = stablehlo.reshape %2826 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %2828 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2829 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2830 = stablehlo.add %arg199, %2829 : tensor<1xi64>
    %2831 = stablehlo.select %2828, %2830, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2832 = stablehlo.reshape %2831 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2833 = stablehlo.reshape %2810 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2834 = stablehlo.transpose %arg324, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2835 = stablehlo.dot %2833, %2834, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2836 = stablehlo.reshape %2835 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2837 = stablehlo.slice %2836 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2838 = stablehlo.reshape %2837 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2839 = stablehlo.slice %2836 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2840 = stablehlo.reshape %2839 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2841 = stablehlo.complex %2838, %2840 : tensor<2x1x32x64xcomplex<f32>>
    %2842 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2843 = stablehlo.multiply %2841, %2842 : tensor<2x1x32x64xcomplex<f32>>
    %2844 = stablehlo.real %2843 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2845 = stablehlo.reshape %2844 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2846 = stablehlo.imag %2843 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2847 = stablehlo.reshape %2846 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2848 = stablehlo.concatenate %2845, %2847, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2849 = stablehlo.reshape %2848 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %2850 = "stablehlo.scatter"(%arg325, %2832, %2849) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2851 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2852 = "stablehlo.gather"(%2850, %2851) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2853 = stablehlo.transpose %2852, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %2854 = stablehlo.reshape %2853 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %2855 = stablehlo.dot_general %2827, %2854, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %2856 = stablehlo.reshape %2855 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %2857 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %2858 = stablehlo.divide %2856, %2857 : tensor<2x32x1x2048xf32>
    %2859 = stablehlo.reduce(%2858 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2860 = stablehlo.broadcast_in_dim %2859, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2861 = stablehlo.subtract %2858, %2860 : tensor<2x32x1x2048xf32>
    %2862 = stablehlo.exponential %2861 : tensor<2x32x1x2048xf32>
    %2863 = stablehlo.reduce(%2862 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2864 = stablehlo.broadcast_in_dim %2863, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2865 = stablehlo.divide %2862, %2864 : tensor<2x32x1x2048xf32>
    %2866 = stablehlo.reshape %2865 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %2867 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2868 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2869 = stablehlo.add %arg199, %2868 : tensor<1xi64>
    %2870 = stablehlo.select %2867, %2869, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2871 = stablehlo.reshape %2870 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2872 = stablehlo.reshape %2810 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2873 = stablehlo.transpose %arg50, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2874 = stablehlo.dot %2872, %2873, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2875 = stablehlo.reshape %2874 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %2876 = "stablehlo.scatter"(%arg323, %2871, %2875) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2877 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2878 = "stablehlo.gather"(%2876, %2877) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2879 = stablehlo.transpose %2878, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %2880 = stablehlo.reshape %2879 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %2881 = stablehlo.dot_general %2866, %2880, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %2882 = stablehlo.reshape %2881 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %2883 = stablehlo.transpose %arg49, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2884 = stablehlo.dot %2882, %2883, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2885 = stablehlo.reshape %2884 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %2886 = stablehlo.add %2798, %2885 : tensor<2x1x4096xf32>
    %2887 = stablehlo.power %2886, %1 : tensor<2x1x4096xf32>
    %2888 = stablehlo.reduce(%2887 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2889 = stablehlo.multiply %2888, %0 : tensor<2x1xf32>
    %2890 = stablehlo.reshape %2889 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %2891 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %2892 = stablehlo.add %2890, %2891 : tensor<2x1x1xf32>
    %2893 = stablehlo.rsqrt %2892 : tensor<2x1x1xf32>
    %2894 = stablehlo.reshape %2893 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %2895 = stablehlo.broadcast_in_dim %2894, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %2896 = stablehlo.multiply %2886, %2895 : tensor<2x1x4096xf32>
    %2897 = stablehlo.broadcast_in_dim %arg48, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %2898 = stablehlo.multiply %2896, %2897 : tensor<2x1x4096xf32>
    %2899 = stablehlo.reshape %2898 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2900 = stablehlo.transpose %arg327, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2901 = stablehlo.dot %2899, %2900, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %2902 = stablehlo.reshape %2901 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %2903 = stablehlo.logistic %2902 : tensor<2x1x11008xf32>
    %2904 = stablehlo.multiply %2902, %2903 : tensor<2x1x11008xf32>
    %2905 = stablehlo.reshape %2898 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2906 = stablehlo.transpose %arg47, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %2907 = stablehlo.dot %2905, %2906, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %2908 = stablehlo.reshape %2907 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %2909 = stablehlo.multiply %2904, %2908 : tensor<2x1x11008xf32>
    %2910 = stablehlo.reshape %2909 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %2911 = stablehlo.transpose %arg46, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %2912 = stablehlo.dot %2910, %2911, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %2913 = stablehlo.reshape %2912 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %2914 = stablehlo.add %2886, %2913 : tensor<2x1x4096xf32>
    %2915 = stablehlo.power %2914, %1 : tensor<2x1x4096xf32>
    %2916 = stablehlo.reduce(%2915 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2917 = stablehlo.multiply %2916, %0 : tensor<2x1xf32>
    %2918 = stablehlo.reshape %2917 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %2919 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %2920 = stablehlo.add %2918, %2919 : tensor<2x1x1xf32>
    %2921 = stablehlo.rsqrt %2920 : tensor<2x1x1xf32>
    %2922 = stablehlo.reshape %2921 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %2923 = stablehlo.broadcast_in_dim %2922, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %2924 = stablehlo.multiply %2914, %2923 : tensor<2x1x4096xf32>
    %2925 = stablehlo.broadcast_in_dim %arg45, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %2926 = stablehlo.multiply %2924, %2925 : tensor<2x1x4096xf32>
    %2927 = stablehlo.reshape %2926 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2928 = stablehlo.transpose %arg331, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2929 = stablehlo.dot %2927, %2928, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2930 = stablehlo.reshape %2929 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2931 = stablehlo.slice %2930 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2932 = stablehlo.reshape %2931 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2933 = stablehlo.slice %2930 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2934 = stablehlo.reshape %2933 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2935 = stablehlo.complex %2932, %2934 : tensor<2x1x32x64xcomplex<f32>>
    %2936 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2937 = stablehlo.multiply %2935, %2936 : tensor<2x1x32x64xcomplex<f32>>
    %2938 = stablehlo.real %2937 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2939 = stablehlo.reshape %2938 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2940 = stablehlo.imag %2937 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2941 = stablehlo.reshape %2940 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2942 = stablehlo.concatenate %2939, %2941, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2943 = stablehlo.reshape %2942 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %2944 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2945 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2946 = stablehlo.add %arg199, %2945 : tensor<1xi64>
    %2947 = stablehlo.select %2944, %2946, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2948 = stablehlo.reshape %2947 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2949 = stablehlo.reshape %2926 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2950 = stablehlo.transpose %arg329, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2951 = stablehlo.dot %2949, %2950, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2952 = stablehlo.reshape %2951 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %2953 = stablehlo.slice %2952 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2954 = stablehlo.reshape %2953 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2955 = stablehlo.slice %2952 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %2956 = stablehlo.reshape %2955 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %2957 = stablehlo.complex %2954, %2956 : tensor<2x1x32x64xcomplex<f32>>
    %2958 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %2959 = stablehlo.multiply %2957, %2958 : tensor<2x1x32x64xcomplex<f32>>
    %2960 = stablehlo.real %2959 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2961 = stablehlo.reshape %2960 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2962 = stablehlo.imag %2959 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %2963 = stablehlo.reshape %2962 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %2964 = stablehlo.concatenate %2961, %2963, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %2965 = stablehlo.reshape %2964 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %2966 = "stablehlo.scatter"(%arg330, %2948, %2965) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2967 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2968 = "stablehlo.gather"(%2966, %2967) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2969 = stablehlo.transpose %2968, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %2970 = stablehlo.reshape %2969 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %2971 = stablehlo.dot_general %2943, %2970, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %2972 = stablehlo.reshape %2971 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %2973 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %2974 = stablehlo.divide %2972, %2973 : tensor<2x32x1x2048xf32>
    %2975 = stablehlo.reduce(%2974 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2976 = stablehlo.broadcast_in_dim %2975, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2977 = stablehlo.subtract %2974, %2976 : tensor<2x32x1x2048xf32>
    %2978 = stablehlo.exponential %2977 : tensor<2x32x1x2048xf32>
    %2979 = stablehlo.reduce(%2978 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %2980 = stablehlo.broadcast_in_dim %2979, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %2981 = stablehlo.divide %2978, %2980 : tensor<2x32x1x2048xf32>
    %2982 = stablehlo.reshape %2981 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %2983 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2984 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %2985 = stablehlo.add %arg199, %2984 : tensor<1xi64>
    %2986 = stablehlo.select %2983, %2985, %arg199 : tensor<1xi1>, tensor<1xi64>
    %2987 = stablehlo.reshape %2986 : (tensor<1xi64>) -> tensor<1x1xi64>
    %2988 = stablehlo.reshape %2926 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %2989 = stablehlo.transpose %arg44, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2990 = stablehlo.dot %2988, %2989, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %2991 = stablehlo.reshape %2990 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %2992 = "stablehlo.scatter"(%arg328, %2987, %2991) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %2993 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %2994 = "stablehlo.gather"(%2992, %2993) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %2995 = stablehlo.transpose %2994, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %2996 = stablehlo.reshape %2995 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %2997 = stablehlo.dot_general %2982, %2996, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %2998 = stablehlo.reshape %2997 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %2999 = stablehlo.transpose %arg43, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3000 = stablehlo.dot %2998, %2999, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3001 = stablehlo.reshape %3000 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %3002 = stablehlo.add %2914, %3001 : tensor<2x1x4096xf32>
    %3003 = stablehlo.power %3002, %1 : tensor<2x1x4096xf32>
    %3004 = stablehlo.reduce(%3003 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3005 = stablehlo.multiply %3004, %0 : tensor<2x1xf32>
    %3006 = stablehlo.reshape %3005 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %3007 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %3008 = stablehlo.add %3006, %3007 : tensor<2x1x1xf32>
    %3009 = stablehlo.rsqrt %3008 : tensor<2x1x1xf32>
    %3010 = stablehlo.reshape %3009 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %3011 = stablehlo.broadcast_in_dim %3010, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %3012 = stablehlo.multiply %3002, %3011 : tensor<2x1x4096xf32>
    %3013 = stablehlo.broadcast_in_dim %arg42, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %3014 = stablehlo.multiply %3012, %3013 : tensor<2x1x4096xf32>
    %3015 = stablehlo.reshape %3014 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3016 = stablehlo.transpose %arg332, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3017 = stablehlo.dot %3015, %3016, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %3018 = stablehlo.reshape %3017 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %3019 = stablehlo.logistic %3018 : tensor<2x1x11008xf32>
    %3020 = stablehlo.multiply %3018, %3019 : tensor<2x1x11008xf32>
    %3021 = stablehlo.reshape %3014 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3022 = stablehlo.transpose %arg41, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3023 = stablehlo.dot %3021, %3022, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %3024 = stablehlo.reshape %3023 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %3025 = stablehlo.multiply %3020, %3024 : tensor<2x1x11008xf32>
    %3026 = stablehlo.reshape %3025 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %3027 = stablehlo.transpose %arg40, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %3028 = stablehlo.dot %3026, %3027, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %3029 = stablehlo.reshape %3028 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %3030 = stablehlo.add %3002, %3029 : tensor<2x1x4096xf32>
    %3031 = stablehlo.power %3030, %1 : tensor<2x1x4096xf32>
    %3032 = stablehlo.reduce(%3031 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3033 = stablehlo.multiply %3032, %0 : tensor<2x1xf32>
    %3034 = stablehlo.reshape %3033 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %3035 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %3036 = stablehlo.add %3034, %3035 : tensor<2x1x1xf32>
    %3037 = stablehlo.rsqrt %3036 : tensor<2x1x1xf32>
    %3038 = stablehlo.reshape %3037 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %3039 = stablehlo.broadcast_in_dim %3038, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %3040 = stablehlo.multiply %3030, %3039 : tensor<2x1x4096xf32>
    %3041 = stablehlo.broadcast_in_dim %arg39, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %3042 = stablehlo.multiply %3040, %3041 : tensor<2x1x4096xf32>
    %3043 = stablehlo.reshape %3042 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3044 = stablehlo.transpose %arg336, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3045 = stablehlo.dot %3043, %3044, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3046 = stablehlo.reshape %3045 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %3047 = stablehlo.slice %3046 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3048 = stablehlo.reshape %3047 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3049 = stablehlo.slice %3046 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3050 = stablehlo.reshape %3049 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3051 = stablehlo.complex %3048, %3050 : tensor<2x1x32x64xcomplex<f32>>
    %3052 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %3053 = stablehlo.multiply %3051, %3052 : tensor<2x1x32x64xcomplex<f32>>
    %3054 = stablehlo.real %3053 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3055 = stablehlo.reshape %3054 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3056 = stablehlo.imag %3053 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3057 = stablehlo.reshape %3056 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3058 = stablehlo.concatenate %3055, %3057, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %3059 = stablehlo.reshape %3058 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %3060 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3061 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %3062 = stablehlo.add %arg199, %3061 : tensor<1xi64>
    %3063 = stablehlo.select %3060, %3062, %arg199 : tensor<1xi1>, tensor<1xi64>
    %3064 = stablehlo.reshape %3063 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3065 = stablehlo.reshape %3042 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3066 = stablehlo.transpose %arg334, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3067 = stablehlo.dot %3065, %3066, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3068 = stablehlo.reshape %3067 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %3069 = stablehlo.slice %3068 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3070 = stablehlo.reshape %3069 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3071 = stablehlo.slice %3068 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3072 = stablehlo.reshape %3071 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3073 = stablehlo.complex %3070, %3072 : tensor<2x1x32x64xcomplex<f32>>
    %3074 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %3075 = stablehlo.multiply %3073, %3074 : tensor<2x1x32x64xcomplex<f32>>
    %3076 = stablehlo.real %3075 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3077 = stablehlo.reshape %3076 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3078 = stablehlo.imag %3075 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3079 = stablehlo.reshape %3078 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3080 = stablehlo.concatenate %3077, %3079, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %3081 = stablehlo.reshape %3080 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %3082 = "stablehlo.scatter"(%arg335, %3064, %3081) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %3083 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %3084 = "stablehlo.gather"(%3082, %3083) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %3085 = stablehlo.transpose %3084, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %3086 = stablehlo.reshape %3085 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %3087 = stablehlo.dot_general %3059, %3086, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %3088 = stablehlo.reshape %3087 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %3089 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %3090 = stablehlo.divide %3088, %3089 : tensor<2x32x1x2048xf32>
    %3091 = stablehlo.reduce(%3090 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3092 = stablehlo.broadcast_in_dim %3091, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %3093 = stablehlo.subtract %3090, %3092 : tensor<2x32x1x2048xf32>
    %3094 = stablehlo.exponential %3093 : tensor<2x32x1x2048xf32>
    %3095 = stablehlo.reduce(%3094 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3096 = stablehlo.broadcast_in_dim %3095, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %3097 = stablehlo.divide %3094, %3096 : tensor<2x32x1x2048xf32>
    %3098 = stablehlo.reshape %3097 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %3099 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3100 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %3101 = stablehlo.add %arg199, %3100 : tensor<1xi64>
    %3102 = stablehlo.select %3099, %3101, %arg199 : tensor<1xi1>, tensor<1xi64>
    %3103 = stablehlo.reshape %3102 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3104 = stablehlo.reshape %3042 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3105 = stablehlo.transpose %arg38, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3106 = stablehlo.dot %3104, %3105, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3107 = stablehlo.reshape %3106 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %3108 = "stablehlo.scatter"(%arg333, %3103, %3107) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %3109 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %3110 = "stablehlo.gather"(%3108, %3109) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %3111 = stablehlo.transpose %3110, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %3112 = stablehlo.reshape %3111 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %3113 = stablehlo.dot_general %3098, %3112, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %3114 = stablehlo.reshape %3113 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %3115 = stablehlo.transpose %arg37, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3116 = stablehlo.dot %3114, %3115, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3117 = stablehlo.reshape %3116 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %3118 = stablehlo.add %3030, %3117 : tensor<2x1x4096xf32>
    %3119 = stablehlo.power %3118, %1 : tensor<2x1x4096xf32>
    %3120 = stablehlo.reduce(%3119 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3121 = stablehlo.multiply %3120, %0 : tensor<2x1xf32>
    %3122 = stablehlo.reshape %3121 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %3123 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %3124 = stablehlo.add %3122, %3123 : tensor<2x1x1xf32>
    %3125 = stablehlo.rsqrt %3124 : tensor<2x1x1xf32>
    %3126 = stablehlo.reshape %3125 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %3127 = stablehlo.broadcast_in_dim %3126, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %3128 = stablehlo.multiply %3118, %3127 : tensor<2x1x4096xf32>
    %3129 = stablehlo.broadcast_in_dim %arg36, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %3130 = stablehlo.multiply %3128, %3129 : tensor<2x1x4096xf32>
    %3131 = stablehlo.reshape %3130 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3132 = stablehlo.transpose %arg337, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3133 = stablehlo.dot %3131, %3132, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %3134 = stablehlo.reshape %3133 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %3135 = stablehlo.logistic %3134 : tensor<2x1x11008xf32>
    %3136 = stablehlo.multiply %3134, %3135 : tensor<2x1x11008xf32>
    %3137 = stablehlo.reshape %3130 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3138 = stablehlo.transpose %arg35, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3139 = stablehlo.dot %3137, %3138, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %3140 = stablehlo.reshape %3139 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %3141 = stablehlo.multiply %3136, %3140 : tensor<2x1x11008xf32>
    %3142 = stablehlo.reshape %3141 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %3143 = stablehlo.transpose %arg34, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %3144 = stablehlo.dot %3142, %3143, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %3145 = stablehlo.reshape %3144 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %3146 = stablehlo.add %3118, %3145 : tensor<2x1x4096xf32>
    %3147 = stablehlo.power %3146, %1 : tensor<2x1x4096xf32>
    %3148 = stablehlo.reduce(%3147 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3149 = stablehlo.multiply %3148, %0 : tensor<2x1xf32>
    %3150 = stablehlo.reshape %3149 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %3151 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %3152 = stablehlo.add %3150, %3151 : tensor<2x1x1xf32>
    %3153 = stablehlo.rsqrt %3152 : tensor<2x1x1xf32>
    %3154 = stablehlo.reshape %3153 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %3155 = stablehlo.broadcast_in_dim %3154, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %3156 = stablehlo.multiply %3146, %3155 : tensor<2x1x4096xf32>
    %3157 = stablehlo.broadcast_in_dim %arg33, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %3158 = stablehlo.multiply %3156, %3157 : tensor<2x1x4096xf32>
    %3159 = stablehlo.reshape %3158 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3160 = stablehlo.transpose %arg341, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3161 = stablehlo.dot %3159, %3160, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3162 = stablehlo.reshape %3161 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %3163 = stablehlo.slice %3162 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3164 = stablehlo.reshape %3163 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3165 = stablehlo.slice %3162 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3166 = stablehlo.reshape %3165 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3167 = stablehlo.complex %3164, %3166 : tensor<2x1x32x64xcomplex<f32>>
    %3168 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %3169 = stablehlo.multiply %3167, %3168 : tensor<2x1x32x64xcomplex<f32>>
    %3170 = stablehlo.real %3169 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3171 = stablehlo.reshape %3170 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3172 = stablehlo.imag %3169 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3173 = stablehlo.reshape %3172 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3174 = stablehlo.concatenate %3171, %3173, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %3175 = stablehlo.reshape %3174 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %3176 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3177 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %3178 = stablehlo.add %arg199, %3177 : tensor<1xi64>
    %3179 = stablehlo.select %3176, %3178, %arg199 : tensor<1xi1>, tensor<1xi64>
    %3180 = stablehlo.reshape %3179 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3181 = stablehlo.reshape %3158 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3182 = stablehlo.transpose %arg339, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3183 = stablehlo.dot %3181, %3182, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3184 = stablehlo.reshape %3183 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %3185 = stablehlo.slice %3184 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3186 = stablehlo.reshape %3185 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3187 = stablehlo.slice %3184 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3188 = stablehlo.reshape %3187 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3189 = stablehlo.complex %3186, %3188 : tensor<2x1x32x64xcomplex<f32>>
    %3190 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %3191 = stablehlo.multiply %3189, %3190 : tensor<2x1x32x64xcomplex<f32>>
    %3192 = stablehlo.real %3191 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3193 = stablehlo.reshape %3192 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3194 = stablehlo.imag %3191 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3195 = stablehlo.reshape %3194 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3196 = stablehlo.concatenate %3193, %3195, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %3197 = stablehlo.reshape %3196 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %3198 = "stablehlo.scatter"(%arg340, %3180, %3197) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %3199 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %3200 = "stablehlo.gather"(%3198, %3199) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %3201 = stablehlo.transpose %3200, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %3202 = stablehlo.reshape %3201 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %3203 = stablehlo.dot_general %3175, %3202, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %3204 = stablehlo.reshape %3203 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %3205 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %3206 = stablehlo.divide %3204, %3205 : tensor<2x32x1x2048xf32>
    %3207 = stablehlo.reduce(%3206 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3208 = stablehlo.broadcast_in_dim %3207, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %3209 = stablehlo.subtract %3206, %3208 : tensor<2x32x1x2048xf32>
    %3210 = stablehlo.exponential %3209 : tensor<2x32x1x2048xf32>
    %3211 = stablehlo.reduce(%3210 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3212 = stablehlo.broadcast_in_dim %3211, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %3213 = stablehlo.divide %3210, %3212 : tensor<2x32x1x2048xf32>
    %3214 = stablehlo.reshape %3213 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %3215 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3216 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %3217 = stablehlo.add %arg199, %3216 : tensor<1xi64>
    %3218 = stablehlo.select %3215, %3217, %arg199 : tensor<1xi1>, tensor<1xi64>
    %3219 = stablehlo.reshape %3218 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3220 = stablehlo.reshape %3158 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3221 = stablehlo.transpose %arg32, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3222 = stablehlo.dot %3220, %3221, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3223 = stablehlo.reshape %3222 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %3224 = "stablehlo.scatter"(%arg338, %3219, %3223) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %3225 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %3226 = "stablehlo.gather"(%3224, %3225) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %3227 = stablehlo.transpose %3226, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %3228 = stablehlo.reshape %3227 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %3229 = stablehlo.dot_general %3214, %3228, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %3230 = stablehlo.reshape %3229 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %3231 = stablehlo.transpose %arg31, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3232 = stablehlo.dot %3230, %3231, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3233 = stablehlo.reshape %3232 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %3234 = stablehlo.add %3146, %3233 : tensor<2x1x4096xf32>
    %3235 = stablehlo.power %3234, %1 : tensor<2x1x4096xf32>
    %3236 = stablehlo.reduce(%3235 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3237 = stablehlo.multiply %3236, %0 : tensor<2x1xf32>
    %3238 = stablehlo.reshape %3237 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %3239 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %3240 = stablehlo.add %3238, %3239 : tensor<2x1x1xf32>
    %3241 = stablehlo.rsqrt %3240 : tensor<2x1x1xf32>
    %3242 = stablehlo.reshape %3241 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %3243 = stablehlo.broadcast_in_dim %3242, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %3244 = stablehlo.multiply %3234, %3243 : tensor<2x1x4096xf32>
    %3245 = stablehlo.broadcast_in_dim %arg30, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %3246 = stablehlo.multiply %3244, %3245 : tensor<2x1x4096xf32>
    %3247 = stablehlo.reshape %3246 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3248 = stablehlo.transpose %arg342, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3249 = stablehlo.dot %3247, %3248, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %3250 = stablehlo.reshape %3249 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %3251 = stablehlo.logistic %3250 : tensor<2x1x11008xf32>
    %3252 = stablehlo.multiply %3250, %3251 : tensor<2x1x11008xf32>
    %3253 = stablehlo.reshape %3246 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3254 = stablehlo.transpose %arg29, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3255 = stablehlo.dot %3253, %3254, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %3256 = stablehlo.reshape %3255 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %3257 = stablehlo.multiply %3252, %3256 : tensor<2x1x11008xf32>
    %3258 = stablehlo.reshape %3257 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %3259 = stablehlo.transpose %arg28, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %3260 = stablehlo.dot %3258, %3259, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %3261 = stablehlo.reshape %3260 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %3262 = stablehlo.add %3234, %3261 : tensor<2x1x4096xf32>
    %3263 = stablehlo.power %3262, %1 : tensor<2x1x4096xf32>
    %3264 = stablehlo.reduce(%3263 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3265 = stablehlo.multiply %3264, %0 : tensor<2x1xf32>
    %3266 = stablehlo.reshape %3265 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %3267 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %3268 = stablehlo.add %3266, %3267 : tensor<2x1x1xf32>
    %3269 = stablehlo.rsqrt %3268 : tensor<2x1x1xf32>
    %3270 = stablehlo.reshape %3269 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %3271 = stablehlo.broadcast_in_dim %3270, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %3272 = stablehlo.multiply %3262, %3271 : tensor<2x1x4096xf32>
    %3273 = stablehlo.broadcast_in_dim %arg27, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %3274 = stablehlo.multiply %3272, %3273 : tensor<2x1x4096xf32>
    %3275 = stablehlo.reshape %3274 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3276 = stablehlo.transpose %arg346, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3277 = stablehlo.dot %3275, %3276, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3278 = stablehlo.reshape %3277 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %3279 = stablehlo.slice %3278 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3280 = stablehlo.reshape %3279 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3281 = stablehlo.slice %3278 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3282 = stablehlo.reshape %3281 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3283 = stablehlo.complex %3280, %3282 : tensor<2x1x32x64xcomplex<f32>>
    %3284 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %3285 = stablehlo.multiply %3283, %3284 : tensor<2x1x32x64xcomplex<f32>>
    %3286 = stablehlo.real %3285 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3287 = stablehlo.reshape %3286 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3288 = stablehlo.imag %3285 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3289 = stablehlo.reshape %3288 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3290 = stablehlo.concatenate %3287, %3289, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %3291 = stablehlo.reshape %3290 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %3292 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3293 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %3294 = stablehlo.add %arg199, %3293 : tensor<1xi64>
    %3295 = stablehlo.select %3292, %3294, %arg199 : tensor<1xi1>, tensor<1xi64>
    %3296 = stablehlo.reshape %3295 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3297 = stablehlo.reshape %3274 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3298 = stablehlo.transpose %arg344, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3299 = stablehlo.dot %3297, %3298, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3300 = stablehlo.reshape %3299 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %3301 = stablehlo.slice %3300 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3302 = stablehlo.reshape %3301 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3303 = stablehlo.slice %3300 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3304 = stablehlo.reshape %3303 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3305 = stablehlo.complex %3302, %3304 : tensor<2x1x32x64xcomplex<f32>>
    %3306 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %3307 = stablehlo.multiply %3305, %3306 : tensor<2x1x32x64xcomplex<f32>>
    %3308 = stablehlo.real %3307 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3309 = stablehlo.reshape %3308 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3310 = stablehlo.imag %3307 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3311 = stablehlo.reshape %3310 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3312 = stablehlo.concatenate %3309, %3311, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %3313 = stablehlo.reshape %3312 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %3314 = "stablehlo.scatter"(%arg345, %3296, %3313) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %3315 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %3316 = "stablehlo.gather"(%3314, %3315) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %3317 = stablehlo.transpose %3316, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %3318 = stablehlo.reshape %3317 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %3319 = stablehlo.dot_general %3291, %3318, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %3320 = stablehlo.reshape %3319 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %3321 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %3322 = stablehlo.divide %3320, %3321 : tensor<2x32x1x2048xf32>
    %3323 = stablehlo.reduce(%3322 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3324 = stablehlo.broadcast_in_dim %3323, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %3325 = stablehlo.subtract %3322, %3324 : tensor<2x32x1x2048xf32>
    %3326 = stablehlo.exponential %3325 : tensor<2x32x1x2048xf32>
    %3327 = stablehlo.reduce(%3326 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3328 = stablehlo.broadcast_in_dim %3327, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %3329 = stablehlo.divide %3326, %3328 : tensor<2x32x1x2048xf32>
    %3330 = stablehlo.reshape %3329 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %3331 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3332 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %3333 = stablehlo.add %arg199, %3332 : tensor<1xi64>
    %3334 = stablehlo.select %3331, %3333, %arg199 : tensor<1xi1>, tensor<1xi64>
    %3335 = stablehlo.reshape %3334 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3336 = stablehlo.reshape %3274 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3337 = stablehlo.transpose %arg26, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3338 = stablehlo.dot %3336, %3337, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3339 = stablehlo.reshape %3338 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %3340 = "stablehlo.scatter"(%arg343, %3335, %3339) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %3341 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %3342 = "stablehlo.gather"(%3340, %3341) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %3343 = stablehlo.transpose %3342, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %3344 = stablehlo.reshape %3343 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %3345 = stablehlo.dot_general %3330, %3344, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %3346 = stablehlo.reshape %3345 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %3347 = stablehlo.transpose %arg25, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3348 = stablehlo.dot %3346, %3347, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3349 = stablehlo.reshape %3348 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %3350 = stablehlo.add %3262, %3349 : tensor<2x1x4096xf32>
    %3351 = stablehlo.power %3350, %1 : tensor<2x1x4096xf32>
    %3352 = stablehlo.reduce(%3351 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3353 = stablehlo.multiply %3352, %0 : tensor<2x1xf32>
    %3354 = stablehlo.reshape %3353 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %3355 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %3356 = stablehlo.add %3354, %3355 : tensor<2x1x1xf32>
    %3357 = stablehlo.rsqrt %3356 : tensor<2x1x1xf32>
    %3358 = stablehlo.reshape %3357 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %3359 = stablehlo.broadcast_in_dim %3358, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %3360 = stablehlo.multiply %3350, %3359 : tensor<2x1x4096xf32>
    %3361 = stablehlo.broadcast_in_dim %arg24, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %3362 = stablehlo.multiply %3360, %3361 : tensor<2x1x4096xf32>
    %3363 = stablehlo.reshape %3362 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3364 = stablehlo.transpose %arg347, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3365 = stablehlo.dot %3363, %3364, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %3366 = stablehlo.reshape %3365 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %3367 = stablehlo.logistic %3366 : tensor<2x1x11008xf32>
    %3368 = stablehlo.multiply %3366, %3367 : tensor<2x1x11008xf32>
    %3369 = stablehlo.reshape %3362 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3370 = stablehlo.transpose %arg23, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3371 = stablehlo.dot %3369, %3370, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %3372 = stablehlo.reshape %3371 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %3373 = stablehlo.multiply %3368, %3372 : tensor<2x1x11008xf32>
    %3374 = stablehlo.reshape %3373 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %3375 = stablehlo.transpose %arg22, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %3376 = stablehlo.dot %3374, %3375, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %3377 = stablehlo.reshape %3376 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %3378 = stablehlo.add %3350, %3377 : tensor<2x1x4096xf32>
    %3379 = stablehlo.power %3378, %1 : tensor<2x1x4096xf32>
    %3380 = stablehlo.reduce(%3379 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3381 = stablehlo.multiply %3380, %0 : tensor<2x1xf32>
    %3382 = stablehlo.reshape %3381 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %3383 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %3384 = stablehlo.add %3382, %3383 : tensor<2x1x1xf32>
    %3385 = stablehlo.rsqrt %3384 : tensor<2x1x1xf32>
    %3386 = stablehlo.reshape %3385 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %3387 = stablehlo.broadcast_in_dim %3386, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %3388 = stablehlo.multiply %3378, %3387 : tensor<2x1x4096xf32>
    %3389 = stablehlo.broadcast_in_dim %arg21, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %3390 = stablehlo.multiply %3388, %3389 : tensor<2x1x4096xf32>
    %3391 = stablehlo.reshape %3390 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3392 = stablehlo.transpose %arg351, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3393 = stablehlo.dot %3391, %3392, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3394 = stablehlo.reshape %3393 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %3395 = stablehlo.slice %3394 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3396 = stablehlo.reshape %3395 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3397 = stablehlo.slice %3394 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3398 = stablehlo.reshape %3397 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3399 = stablehlo.complex %3396, %3398 : tensor<2x1x32x64xcomplex<f32>>
    %3400 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %3401 = stablehlo.multiply %3399, %3400 : tensor<2x1x32x64xcomplex<f32>>
    %3402 = stablehlo.real %3401 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3403 = stablehlo.reshape %3402 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3404 = stablehlo.imag %3401 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3405 = stablehlo.reshape %3404 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3406 = stablehlo.concatenate %3403, %3405, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %3407 = stablehlo.reshape %3406 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %3408 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3409 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %3410 = stablehlo.add %arg199, %3409 : tensor<1xi64>
    %3411 = stablehlo.select %3408, %3410, %arg199 : tensor<1xi1>, tensor<1xi64>
    %3412 = stablehlo.reshape %3411 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3413 = stablehlo.reshape %3390 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3414 = stablehlo.transpose %arg349, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3415 = stablehlo.dot %3413, %3414, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3416 = stablehlo.reshape %3415 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %3417 = stablehlo.slice %3416 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3418 = stablehlo.reshape %3417 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3419 = stablehlo.slice %3416 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3420 = stablehlo.reshape %3419 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3421 = stablehlo.complex %3418, %3420 : tensor<2x1x32x64xcomplex<f32>>
    %3422 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %3423 = stablehlo.multiply %3421, %3422 : tensor<2x1x32x64xcomplex<f32>>
    %3424 = stablehlo.real %3423 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3425 = stablehlo.reshape %3424 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3426 = stablehlo.imag %3423 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3427 = stablehlo.reshape %3426 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3428 = stablehlo.concatenate %3425, %3427, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %3429 = stablehlo.reshape %3428 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %3430 = "stablehlo.scatter"(%arg350, %3412, %3429) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %3431 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %3432 = "stablehlo.gather"(%3430, %3431) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %3433 = stablehlo.transpose %3432, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %3434 = stablehlo.reshape %3433 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %3435 = stablehlo.dot_general %3407, %3434, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %3436 = stablehlo.reshape %3435 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %3437 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %3438 = stablehlo.divide %3436, %3437 : tensor<2x32x1x2048xf32>
    %3439 = stablehlo.reduce(%3438 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3440 = stablehlo.broadcast_in_dim %3439, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %3441 = stablehlo.subtract %3438, %3440 : tensor<2x32x1x2048xf32>
    %3442 = stablehlo.exponential %3441 : tensor<2x32x1x2048xf32>
    %3443 = stablehlo.reduce(%3442 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3444 = stablehlo.broadcast_in_dim %3443, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %3445 = stablehlo.divide %3442, %3444 : tensor<2x32x1x2048xf32>
    %3446 = stablehlo.reshape %3445 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %3447 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3448 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %3449 = stablehlo.add %arg199, %3448 : tensor<1xi64>
    %3450 = stablehlo.select %3447, %3449, %arg199 : tensor<1xi1>, tensor<1xi64>
    %3451 = stablehlo.reshape %3450 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3452 = stablehlo.reshape %3390 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3453 = stablehlo.transpose %arg20, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3454 = stablehlo.dot %3452, %3453, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3455 = stablehlo.reshape %3454 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %3456 = "stablehlo.scatter"(%arg348, %3451, %3455) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %3457 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %3458 = "stablehlo.gather"(%3456, %3457) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %3459 = stablehlo.transpose %3458, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %3460 = stablehlo.reshape %3459 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %3461 = stablehlo.dot_general %3446, %3460, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %3462 = stablehlo.reshape %3461 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %3463 = stablehlo.transpose %arg19, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3464 = stablehlo.dot %3462, %3463, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3465 = stablehlo.reshape %3464 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %3466 = stablehlo.add %3378, %3465 : tensor<2x1x4096xf32>
    %3467 = stablehlo.power %3466, %1 : tensor<2x1x4096xf32>
    %3468 = stablehlo.reduce(%3467 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3469 = stablehlo.multiply %3468, %0 : tensor<2x1xf32>
    %3470 = stablehlo.reshape %3469 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %3471 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %3472 = stablehlo.add %3470, %3471 : tensor<2x1x1xf32>
    %3473 = stablehlo.rsqrt %3472 : tensor<2x1x1xf32>
    %3474 = stablehlo.reshape %3473 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %3475 = stablehlo.broadcast_in_dim %3474, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %3476 = stablehlo.multiply %3466, %3475 : tensor<2x1x4096xf32>
    %3477 = stablehlo.broadcast_in_dim %arg18, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %3478 = stablehlo.multiply %3476, %3477 : tensor<2x1x4096xf32>
    %3479 = stablehlo.reshape %3478 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3480 = stablehlo.transpose %arg352, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3481 = stablehlo.dot %3479, %3480, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %3482 = stablehlo.reshape %3481 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %3483 = stablehlo.logistic %3482 : tensor<2x1x11008xf32>
    %3484 = stablehlo.multiply %3482, %3483 : tensor<2x1x11008xf32>
    %3485 = stablehlo.reshape %3478 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3486 = stablehlo.transpose %arg17, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3487 = stablehlo.dot %3485, %3486, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %3488 = stablehlo.reshape %3487 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %3489 = stablehlo.multiply %3484, %3488 : tensor<2x1x11008xf32>
    %3490 = stablehlo.reshape %3489 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %3491 = stablehlo.transpose %arg16, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %3492 = stablehlo.dot %3490, %3491, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %3493 = stablehlo.reshape %3492 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %3494 = stablehlo.add %3466, %3493 : tensor<2x1x4096xf32>
    %3495 = stablehlo.power %3494, %1 : tensor<2x1x4096xf32>
    %3496 = stablehlo.reduce(%3495 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3497 = stablehlo.multiply %3496, %0 : tensor<2x1xf32>
    %3498 = stablehlo.reshape %3497 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %3499 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %3500 = stablehlo.add %3498, %3499 : tensor<2x1x1xf32>
    %3501 = stablehlo.rsqrt %3500 : tensor<2x1x1xf32>
    %3502 = stablehlo.reshape %3501 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %3503 = stablehlo.broadcast_in_dim %3502, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %3504 = stablehlo.multiply %3494, %3503 : tensor<2x1x4096xf32>
    %3505 = stablehlo.broadcast_in_dim %arg15, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %3506 = stablehlo.multiply %3504, %3505 : tensor<2x1x4096xf32>
    %3507 = stablehlo.reshape %3506 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3508 = stablehlo.transpose %arg356, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3509 = stablehlo.dot %3507, %3508, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3510 = stablehlo.reshape %3509 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %3511 = stablehlo.slice %3510 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3512 = stablehlo.reshape %3511 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3513 = stablehlo.slice %3510 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3514 = stablehlo.reshape %3513 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3515 = stablehlo.complex %3512, %3514 : tensor<2x1x32x64xcomplex<f32>>
    %3516 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %3517 = stablehlo.multiply %3515, %3516 : tensor<2x1x32x64xcomplex<f32>>
    %3518 = stablehlo.real %3517 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3519 = stablehlo.reshape %3518 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3520 = stablehlo.imag %3517 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3521 = stablehlo.reshape %3520 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3522 = stablehlo.concatenate %3519, %3521, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %3523 = stablehlo.reshape %3522 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %3524 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3525 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %3526 = stablehlo.add %arg199, %3525 : tensor<1xi64>
    %3527 = stablehlo.select %3524, %3526, %arg199 : tensor<1xi1>, tensor<1xi64>
    %3528 = stablehlo.reshape %3527 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3529 = stablehlo.reshape %3506 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3530 = stablehlo.transpose %arg354, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3531 = stablehlo.dot %3529, %3530, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3532 = stablehlo.reshape %3531 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %3533 = stablehlo.slice %3532 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3534 = stablehlo.reshape %3533 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3535 = stablehlo.slice %3532 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3536 = stablehlo.reshape %3535 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3537 = stablehlo.complex %3534, %3536 : tensor<2x1x32x64xcomplex<f32>>
    %3538 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %3539 = stablehlo.multiply %3537, %3538 : tensor<2x1x32x64xcomplex<f32>>
    %3540 = stablehlo.real %3539 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3541 = stablehlo.reshape %3540 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3542 = stablehlo.imag %3539 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3543 = stablehlo.reshape %3542 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3544 = stablehlo.concatenate %3541, %3543, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %3545 = stablehlo.reshape %3544 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %3546 = "stablehlo.scatter"(%arg355, %3528, %3545) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %3547 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %3548 = "stablehlo.gather"(%3546, %3547) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %3549 = stablehlo.transpose %3548, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %3550 = stablehlo.reshape %3549 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %3551 = stablehlo.dot_general %3523, %3550, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %3552 = stablehlo.reshape %3551 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %3553 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %3554 = stablehlo.divide %3552, %3553 : tensor<2x32x1x2048xf32>
    %3555 = stablehlo.reduce(%3554 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3556 = stablehlo.broadcast_in_dim %3555, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %3557 = stablehlo.subtract %3554, %3556 : tensor<2x32x1x2048xf32>
    %3558 = stablehlo.exponential %3557 : tensor<2x32x1x2048xf32>
    %3559 = stablehlo.reduce(%3558 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3560 = stablehlo.broadcast_in_dim %3559, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %3561 = stablehlo.divide %3558, %3560 : tensor<2x32x1x2048xf32>
    %3562 = stablehlo.reshape %3561 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %3563 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3564 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %3565 = stablehlo.add %arg199, %3564 : tensor<1xi64>
    %3566 = stablehlo.select %3563, %3565, %arg199 : tensor<1xi1>, tensor<1xi64>
    %3567 = stablehlo.reshape %3566 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3568 = stablehlo.reshape %3506 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3569 = stablehlo.transpose %arg14, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3570 = stablehlo.dot %3568, %3569, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3571 = stablehlo.reshape %3570 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %3572 = "stablehlo.scatter"(%arg353, %3567, %3571) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %3573 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %3574 = "stablehlo.gather"(%3572, %3573) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %3575 = stablehlo.transpose %3574, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %3576 = stablehlo.reshape %3575 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %3577 = stablehlo.dot_general %3562, %3576, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %3578 = stablehlo.reshape %3577 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %3579 = stablehlo.transpose %arg13, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3580 = stablehlo.dot %3578, %3579, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3581 = stablehlo.reshape %3580 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %3582 = stablehlo.add %3494, %3581 : tensor<2x1x4096xf32>
    %3583 = stablehlo.power %3582, %1 : tensor<2x1x4096xf32>
    %3584 = stablehlo.reduce(%3583 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3585 = stablehlo.multiply %3584, %0 : tensor<2x1xf32>
    %3586 = stablehlo.reshape %3585 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %3587 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %3588 = stablehlo.add %3586, %3587 : tensor<2x1x1xf32>
    %3589 = stablehlo.rsqrt %3588 : tensor<2x1x1xf32>
    %3590 = stablehlo.reshape %3589 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %3591 = stablehlo.broadcast_in_dim %3590, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %3592 = stablehlo.multiply %3582, %3591 : tensor<2x1x4096xf32>
    %3593 = stablehlo.broadcast_in_dim %arg12, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %3594 = stablehlo.multiply %3592, %3593 : tensor<2x1x4096xf32>
    %3595 = stablehlo.reshape %3594 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3596 = stablehlo.transpose %arg357, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3597 = stablehlo.dot %3595, %3596, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %3598 = stablehlo.reshape %3597 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %3599 = stablehlo.logistic %3598 : tensor<2x1x11008xf32>
    %3600 = stablehlo.multiply %3598, %3599 : tensor<2x1x11008xf32>
    %3601 = stablehlo.reshape %3594 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3602 = stablehlo.transpose %arg11, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3603 = stablehlo.dot %3601, %3602, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %3604 = stablehlo.reshape %3603 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %3605 = stablehlo.multiply %3600, %3604 : tensor<2x1x11008xf32>
    %3606 = stablehlo.reshape %3605 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %3607 = stablehlo.transpose %arg10, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %3608 = stablehlo.dot %3606, %3607, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %3609 = stablehlo.reshape %3608 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %3610 = stablehlo.add %3582, %3609 : tensor<2x1x4096xf32>
    %3611 = stablehlo.power %3610, %1 : tensor<2x1x4096xf32>
    %3612 = stablehlo.reduce(%3611 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3613 = stablehlo.multiply %3612, %0 : tensor<2x1xf32>
    %3614 = stablehlo.reshape %3613 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %3615 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %3616 = stablehlo.add %3614, %3615 : tensor<2x1x1xf32>
    %3617 = stablehlo.rsqrt %3616 : tensor<2x1x1xf32>
    %3618 = stablehlo.reshape %3617 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %3619 = stablehlo.broadcast_in_dim %3618, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %3620 = stablehlo.multiply %3610, %3619 : tensor<2x1x4096xf32>
    %3621 = stablehlo.broadcast_in_dim %arg9, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %3622 = stablehlo.multiply %3620, %3621 : tensor<2x1x4096xf32>
    %3623 = stablehlo.reshape %3622 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3624 = stablehlo.transpose %arg361, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3625 = stablehlo.dot %3623, %3624, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3626 = stablehlo.reshape %3625 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %3627 = stablehlo.slice %3626 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3628 = stablehlo.reshape %3627 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3629 = stablehlo.slice %3626 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3630 = stablehlo.reshape %3629 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3631 = stablehlo.complex %3628, %3630 : tensor<2x1x32x64xcomplex<f32>>
    %3632 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %3633 = stablehlo.multiply %3631, %3632 : tensor<2x1x32x64xcomplex<f32>>
    %3634 = stablehlo.real %3633 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3635 = stablehlo.reshape %3634 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3636 = stablehlo.imag %3633 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3637 = stablehlo.reshape %3636 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3638 = stablehlo.concatenate %3635, %3637, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %3639 = stablehlo.reshape %3638 : (tensor<2x1x32x64x2xf32>) -> tensor<64x1x128xf32>
    %3640 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3641 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %3642 = stablehlo.add %arg199, %3641 : tensor<1xi64>
    %3643 = stablehlo.select %3640, %3642, %arg199 : tensor<1xi1>, tensor<1xi64>
    %3644 = stablehlo.reshape %3643 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3645 = stablehlo.reshape %3622 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3646 = stablehlo.transpose %arg359, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3647 = stablehlo.dot %3645, %3646, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3648 = stablehlo.reshape %3647 : (tensor<2x4096xf32>) -> tensor<2x1x32x64x2xf32>
    %3649 = stablehlo.slice %3648 [0:2, 0:1, 0:32, 0:64, 0:1] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3650 = stablehlo.reshape %3649 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3651 = stablehlo.slice %3648 [0:2, 0:1, 0:32, 0:64, 1:2] : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x64x1xf32>
    %3652 = stablehlo.reshape %3651 : (tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64xf32>
    %3653 = stablehlo.complex %3650, %3652 : tensor<2x1x32x64xcomplex<f32>>
    %3654 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x64xcomplex<f32>>) -> tensor<2x1x32x64xcomplex<f32>>
    %3655 = stablehlo.multiply %3653, %3654 : tensor<2x1x32x64xcomplex<f32>>
    %3656 = stablehlo.real %3655 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3657 = stablehlo.reshape %3656 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3658 = stablehlo.imag %3655 : (tensor<2x1x32x64xcomplex<f32>>) -> tensor<2x1x32x64xf32>
    %3659 = stablehlo.reshape %3658 : (tensor<2x1x32x64xf32>) -> tensor<2x1x32x64x1xf32>
    %3660 = stablehlo.concatenate %3657, %3659, dim = 4 : (tensor<2x1x32x64x1xf32>, tensor<2x1x32x64x1xf32>) -> tensor<2x1x32x64x2xf32>
    %3661 = stablehlo.reshape %3660 : (tensor<2x1x32x64x2xf32>) -> tensor<2x1x32x128xf32>
    %3662 = "stablehlo.scatter"(%arg360, %3644, %3661) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %3663 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %3664 = "stablehlo.gather"(%3662, %3663) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %3665 = stablehlo.transpose %3664, dims = [0, 2, 3, 1] : (tensor<2x2048x32x128xf32>) -> tensor<2x32x128x2048xf32>
    %3666 = stablehlo.reshape %3665 : (tensor<2x32x128x2048xf32>) -> tensor<64x128x2048xf32>
    %3667 = stablehlo.dot_general %3639, %3666, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x128xf32>, tensor<64x128x2048xf32>) -> tensor<64x1x2048xf32>
    %3668 = stablehlo.reshape %3667 : (tensor<64x1x2048xf32>) -> tensor<2x32x1x2048xf32>
    %3669 = stablehlo.broadcast_in_dim %arg202, dims = [] : (tensor<f32>) -> tensor<2x32x1x2048xf32>
    %3670 = stablehlo.divide %3668, %3669 : tensor<2x32x1x2048xf32>
    %3671 = stablehlo.reduce(%3670 init: %4) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.maximum %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3672 = stablehlo.broadcast_in_dim %3671, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %3673 = stablehlo.subtract %3670, %3672 : tensor<2x32x1x2048xf32>
    %3674 = stablehlo.exponential %3673 : tensor<2x32x1x2048xf32>
    %3675 = stablehlo.reduce(%3674 init: %5) across dimensions = [3] : (tensor<2x32x1x2048xf32>, tensor<f32>) -> tensor<2x32x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3676 = stablehlo.broadcast_in_dim %3675, dims = [0, 1, 2] : (tensor<2x32x1xf32>) -> tensor<2x32x1x2048xf32>
    %3677 = stablehlo.divide %3674, %3676 : tensor<2x32x1x2048xf32>
    %3678 = stablehlo.reshape %3677 : (tensor<2x32x1x2048xf32>) -> tensor<64x1x2048xf32>
    %3679 = stablehlo.compare  LT, %arg199, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %3680 = stablehlo.reshape %arg200 : (tensor<i64>) -> tensor<1xi64>
    %3681 = stablehlo.add %arg199, %3680 : tensor<1xi64>
    %3682 = stablehlo.select %3679, %3681, %arg199 : tensor<1xi1>, tensor<1xi64>
    %3683 = stablehlo.reshape %3682 : (tensor<1xi64>) -> tensor<1x1xi64>
    %3684 = stablehlo.reshape %3622 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3685 = stablehlo.transpose %arg8, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3686 = stablehlo.dot %3684, %3685, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3687 = stablehlo.reshape %3686 : (tensor<2x4096xf32>) -> tensor<2x1x32x128xf32>
    %3688 = "stablehlo.scatter"(%arg358, %3683, %3687) ({
    ^bb0(%arg363: tensor<f32>, %arg364: tensor<f32>):
      stablehlo.return %arg364 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x32x128xf32>, tensor<1x1xi64>, tensor<2x1x32x128xf32>) -> tensor<2x2304x32x128xf32>
    %3689 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %3690 = "stablehlo.gather"(%3688, %3689) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 32, 128]> : tensor<4xi64>} : (tensor<2x2304x32x128xf32>, tensor<2048xui32>) -> tensor<2x2048x32x128xf32>
    %3691 = stablehlo.transpose %3690, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,32,2048,128]{3,1,2,0}"} : (tensor<2x2048x32x128xf32>) -> tensor<2x32x2048x128xf32>
    %3692 = stablehlo.reshape %3691 : (tensor<2x32x2048x128xf32>) -> tensor<64x2048x128xf32>
    %3693 = stablehlo.dot_general %3678, %3692, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x2048xf32>, tensor<64x2048x128xf32>) -> tensor<64x1x128xf32>
    %3694 = stablehlo.reshape %3693 : (tensor<64x1x128xf32>) -> tensor<2x4096xf32>
    %3695 = stablehlo.transpose %arg6, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,4096]{0,1}"} : (tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3696 = stablehlo.dot %3694, %3695, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x4096xf32>) -> tensor<2x4096xf32>
    %3697 = stablehlo.reshape %3696 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %3698 = stablehlo.add %3610, %3697 : tensor<2x1x4096xf32>
    %3699 = stablehlo.power %3698, %1 : tensor<2x1x4096xf32>
    %3700 = stablehlo.reduce(%3699 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3701 = stablehlo.multiply %3700, %0 : tensor<2x1xf32>
    %3702 = stablehlo.reshape %3701 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %3703 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %3704 = stablehlo.add %3702, %3703 : tensor<2x1x1xf32>
    %3705 = stablehlo.rsqrt %3704 : tensor<2x1x1xf32>
    %3706 = stablehlo.reshape %3705 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %3707 = stablehlo.broadcast_in_dim %3706, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %3708 = stablehlo.multiply %3698, %3707 : tensor<2x1x4096xf32>
    %3709 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %3710 = stablehlo.multiply %3708, %3709 : tensor<2x1x4096xf32>
    %3711 = stablehlo.reshape %3710 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3712 = stablehlo.transpose %arg362, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3713 = stablehlo.dot %3711, %3712, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %3714 = stablehlo.reshape %3713 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %3715 = stablehlo.logistic %3714 : tensor<2x1x11008xf32>
    %3716 = stablehlo.multiply %3714, %3715 : tensor<2x1x11008xf32>
    %3717 = stablehlo.reshape %3710 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3718 = stablehlo.transpose %arg4, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,11008]{0,1}"} : (tensor<11008x4096xf32>) -> tensor<4096x11008xf32>
    %3719 = stablehlo.dot %3717, %3718, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x11008xf32>) -> tensor<2x11008xf32>
    %3720 = stablehlo.reshape %3719 : (tensor<2x11008xf32>) -> tensor<2x1x11008xf32>
    %3721 = stablehlo.multiply %3716, %3720 : tensor<2x1x11008xf32>
    %3722 = stablehlo.reshape %3721 : (tensor<2x1x11008xf32>) -> tensor<2x11008xf32>
    %3723 = stablehlo.transpose %arg3, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[11008,4096]{0,1}"} : (tensor<4096x11008xf32>) -> tensor<11008x4096xf32>
    %3724 = stablehlo.dot %3722, %3723, precision = [DEFAULT, DEFAULT] : (tensor<2x11008xf32>, tensor<11008x4096xf32>) -> tensor<2x4096xf32>
    %3725 = stablehlo.reshape %3724 : (tensor<2x4096xf32>) -> tensor<2x1x4096xf32>
    %3726 = stablehlo.add %3698, %3725 : tensor<2x1x4096xf32>
    %3727 = stablehlo.power %3726, %1 : tensor<2x1x4096xf32>
    %3728 = stablehlo.reduce(%3727 init: %5) across dimensions = [2] : (tensor<2x1x4096xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg363: tensor<f32>, %arg364: tensor<f32>)  {
      %3743 = stablehlo.add %arg363, %arg364 : tensor<f32>
      stablehlo.return %3743 : tensor<f32>
    }
    %3729 = stablehlo.multiply %3728, %0 : tensor<2x1xf32>
    %3730 = stablehlo.reshape %3729 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %3731 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %3732 = stablehlo.add %3730, %3731 : tensor<2x1x1xf32>
    %3733 = stablehlo.rsqrt %3732 : tensor<2x1x1xf32>
    %3734 = stablehlo.reshape %3733 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %3735 = stablehlo.broadcast_in_dim %3734, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x4096xf32>
    %3736 = stablehlo.multiply %3726, %3735 : tensor<2x1x4096xf32>
    %3737 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<4096xf32>) -> tensor<2x1x4096xf32>
    %3738 = stablehlo.multiply %3736, %3737 : tensor<2x1x4096xf32>
    %3739 = stablehlo.reshape %3738 : (tensor<2x1x4096xf32>) -> tensor<2x4096xf32>
    %3740 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[4096,32000]{0,1}"} : (tensor<32000x4096xf32>) -> tensor<4096x32000xf32>
    %3741 = stablehlo.dot %3739, %3740, precision = [DEFAULT, DEFAULT] : (tensor<2x4096xf32>, tensor<4096x32000xf32>) -> tensor<2x32000xf32>
    %3742 = stablehlo.reshape %3741 : (tensor<2x32000xf32>) -> tensor<2x1x32000xf32>
    return %3742, %66, %92, %182, %208, %298, %324, %414, %440, %530, %556, %646, %672, %762, %788, %878, %904, %994, %1020, %1110, %1136, %1226, %1252, %1342, %1368, %1458, %1484, %1574, %1600, %1690, %1716, %1806, %1832, %1922, %1948, %2038, %2064, %2154, %2180, %2270, %2296, %2386, %2412, %2502, %2528, %2618, %2644, %2734, %2760, %2850, %2876, %2966, %2992, %3082, %3108, %3198, %3224, %3314, %3340, %3430, %3456, %3546, %3572, %3662, %3688 : tensor<2x1x32000xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>, tensor<2x2304x32x128xf32>
  }
}
