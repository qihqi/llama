module @IrToHlo.765 attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<32000x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<f32>, %arg3: tensor<128x352xf32>, %arg4: tensor<352x128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<128x128xf32>, %arg7: tensor<2048xi64>, %arg8: tensor<128x128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<128x352xf32>, %arg11: tensor<352x128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<128x128xf32>, %arg14: tensor<128x128xf32>, %arg15: tensor<128xf32>, %arg16: tensor<128x352xf32>, %arg17: tensor<352x128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<128x128xf32>, %arg20: tensor<128x128xf32>, %arg21: tensor<128xf32>, %arg22: tensor<2x1xi64>, %arg23: tensor<i64>, %arg24: tensor<32000x128xf32>, %arg25: tensor<1xi64>, %arg26: tensor<i64>, %arg27: tensor<2x2304x2x64xf32>, %arg28: tensor<f32>, %arg29: tensor<4608x32xcomplex<f32>>, %arg30: tensor<128x128xf32>, %arg31: tensor<2x2304x2x64xf32>, %arg32: tensor<128x128xf32>, %arg33: tensor<352x128xf32>, %arg34: tensor<2x2304x2x64xf32>, %arg35: tensor<128x128xf32>, %arg36: tensor<2x2304x2x64xf32>, %arg37: tensor<128x128xf32>, %arg38: tensor<352x128xf32>, %arg39: tensor<2x2304x2x64xf32>, %arg40: tensor<128x128xf32>, %arg41: tensor<2x2304x2x64xf32>, %arg42: tensor<128x128xf32>, %arg43: tensor<352x128xf32>) -> (tensor<2x1x32000xf32>, tensor<2x2304x2x64xf32>, tensor<2x2304x2x64xf32>, tensor<2x2304x2x64xf32>, tensor<2x2304x2x64xf32>, tensor<2x2304x2x64xf32>, tensor<2x2304x2x64xf32>) {
    %0 = stablehlo.constant dense<7.812500e-03> : tensor<2x1xf32>
    %1 = stablehlo.constant dense<2.000000e+00> : tensor<2x1x128xf32>
    %2 = stablehlo.constant dense<0> : tensor<1xi64>
    %3 = stablehlo.constant dense<0> : tensor<2x1xi64>
    %4 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.compare  LT, %arg22, %3 : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x1xi1>
    %7 = stablehlo.reshape %arg23 : (tensor<i64>) -> tensor<1xi64>
    %8 = stablehlo.broadcast_in_dim %7, dims = [1] : (tensor<1xi64>) -> tensor<2x1xi64>
    %9 = stablehlo.add %arg22, %8 : tensor<2x1xi64>
    %10 = stablehlo.select %6, %9, %arg22 : tensor<2x1xi1>, tensor<2x1xi64>
    %11 = stablehlo.reshape %10 : (tensor<2x1xi64>) -> tensor<2x1x1xi64>
    %12 = "stablehlo.gather"(%arg24, %11) {dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<32000x128xf32>, tensor<2x1x1xi64>) -> tensor<2x1x128xf32>
    %13 = stablehlo.power %12, %1 : tensor<2x1x128xf32>
    %14 = stablehlo.reduce(%13 init: %5) across dimensions = [2] : (tensor<2x1x128xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg44: tensor<f32>, %arg45: tensor<f32>)  {
      %379 = stablehlo.add %arg44, %arg45 : tensor<f32>
      stablehlo.return %379 : tensor<f32>
    }
    %15 = stablehlo.multiply %14, %0 : tensor<2x1xf32>
    %16 = stablehlo.reshape %15 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %17 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %18 = stablehlo.add %16, %17 : tensor<2x1x1xf32>
    %19 = stablehlo.rsqrt %18 : tensor<2x1x1xf32>
    %20 = stablehlo.reshape %19 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x128xf32>
    %22 = stablehlo.multiply %12, %21 : tensor<2x1x128xf32>
    %23 = stablehlo.broadcast_in_dim %arg21, dims = [2] : (tensor<128xf32>) -> tensor<2x1x128xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<2x1x128xf32>
    %25 = stablehlo.reshape %24 : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %26 = stablehlo.transpose %arg32, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %27 = stablehlo.dot %25, %26, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %28 = stablehlo.reshape %27 : (tensor<2x128xf32>) -> tensor<2x1x2x32x2xf32>
    %29 = stablehlo.slice %28 [0:2, 0:1, 0:2, 0:32, 0:1] : (tensor<2x1x2x32x2xf32>) -> tensor<2x1x2x32x1xf32>
    %30 = stablehlo.reshape %29 : (tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32xf32>
    %31 = stablehlo.slice %28 [0:2, 0:1, 0:2, 0:32, 1:2] : (tensor<2x1x2x32x2xf32>) -> tensor<2x1x2x32x1xf32>
    %32 = stablehlo.reshape %31 : (tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32xf32>
    %33 = stablehlo.complex %30, %32 : tensor<2x1x2x32xcomplex<f32>>
    %34 = stablehlo.convert %arg25 : (tensor<1xi64>) -> tensor<1xui32>
    %35 = "stablehlo.gather"(%arg29, %34) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32]> : tensor<2xi64>} : (tensor<4608x32xcomplex<f32>>, tensor<1xui32>) -> tensor<1x32xcomplex<f32>>
    %36 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x32xcomplex<f32>>) -> tensor<2x1x2x32xcomplex<f32>>
    %37 = stablehlo.multiply %33, %36 : tensor<2x1x2x32xcomplex<f32>>
    %38 = stablehlo.real %37 : (tensor<2x1x2x32xcomplex<f32>>) -> tensor<2x1x2x32xf32>
    %39 = stablehlo.reshape %38 : (tensor<2x1x2x32xf32>) -> tensor<2x1x2x32x1xf32>
    %40 = stablehlo.imag %37 : (tensor<2x1x2x32xcomplex<f32>>) -> tensor<2x1x2x32xf32>
    %41 = stablehlo.reshape %40 : (tensor<2x1x2x32xf32>) -> tensor<2x1x2x32x1xf32>
    %42 = stablehlo.concatenate %39, %41, dim = 4 : (tensor<2x1x2x32x1xf32>, tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32x2xf32>
    %43 = stablehlo.reshape %42 : (tensor<2x1x2x32x2xf32>) -> tensor<4x1x64xf32>
    %44 = stablehlo.compare  LT, %arg25, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %45 = stablehlo.reshape %arg26 : (tensor<i64>) -> tensor<1xi64>
    %46 = stablehlo.add %arg25, %45 : tensor<1xi64>
    %47 = stablehlo.select %44, %46, %arg25 : tensor<1xi1>, tensor<1xi64>
    %48 = stablehlo.reshape %47 : (tensor<1xi64>) -> tensor<1x1xi64>
    %49 = stablehlo.reshape %24 : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %50 = stablehlo.transpose %arg30, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %51 = stablehlo.dot %49, %50, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %52 = stablehlo.reshape %51 : (tensor<2x128xf32>) -> tensor<2x1x2x32x2xf32>
    %53 = stablehlo.slice %52 [0:2, 0:1, 0:2, 0:32, 0:1] : (tensor<2x1x2x32x2xf32>) -> tensor<2x1x2x32x1xf32>
    %54 = stablehlo.reshape %53 : (tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32xf32>
    %55 = stablehlo.slice %52 [0:2, 0:1, 0:2, 0:32, 1:2] : (tensor<2x1x2x32x2xf32>) -> tensor<2x1x2x32x1xf32>
    %56 = stablehlo.reshape %55 : (tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32xf32>
    %57 = stablehlo.complex %54, %56 : tensor<2x1x2x32xcomplex<f32>>
    %58 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x32xcomplex<f32>>) -> tensor<2x1x2x32xcomplex<f32>>
    %59 = stablehlo.multiply %57, %58 : tensor<2x1x2x32xcomplex<f32>>
    %60 = stablehlo.real %59 : (tensor<2x1x2x32xcomplex<f32>>) -> tensor<2x1x2x32xf32>
    %61 = stablehlo.reshape %60 : (tensor<2x1x2x32xf32>) -> tensor<2x1x2x32x1xf32>
    %62 = stablehlo.imag %59 : (tensor<2x1x2x32xcomplex<f32>>) -> tensor<2x1x2x32xf32>
    %63 = stablehlo.reshape %62 : (tensor<2x1x2x32xf32>) -> tensor<2x1x2x32x1xf32>
    %64 = stablehlo.concatenate %61, %63, dim = 4 : (tensor<2x1x2x32x1xf32>, tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32x2xf32>
    %65 = stablehlo.reshape %64 : (tensor<2x1x2x32x2xf32>) -> tensor<2x1x2x64xf32>
    %66 = "stablehlo.scatter"(%arg31, %48, %65) ({
    ^bb0(%arg44: tensor<f32>, %arg45: tensor<f32>):
      stablehlo.return %arg45 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x2x64xf32>, tensor<1x1xi64>, tensor<2x1x2x64xf32>) -> tensor<2x2304x2x64xf32>
    %67 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %68 = "stablehlo.gather"(%66, %67) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 2, 64]> : tensor<4xi64>} : (tensor<2x2304x2x64xf32>, tensor<2048xui32>) -> tensor<2x2048x2x64xf32>
    %69 = stablehlo.transpose %68, dims = [0, 2, 3, 1] : (tensor<2x2048x2x64xf32>) -> tensor<2x2x64x2048xf32>
    %70 = stablehlo.reshape %69 : (tensor<2x2x64x2048xf32>) -> tensor<4x64x2048xf32>
    %71 = stablehlo.dot_general %43, %70, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x1x64xf32>, tensor<4x64x2048xf32>) -> tensor<4x1x2048xf32>
    %72 = stablehlo.reshape %71 : (tensor<4x1x2048xf32>) -> tensor<2x2x1x2048xf32>
    %73 = stablehlo.broadcast_in_dim %arg28, dims = [] : (tensor<f32>) -> tensor<2x2x1x2048xf32>
    %74 = stablehlo.divide %72, %73 : tensor<2x2x1x2048xf32>
    %75 = stablehlo.reduce(%74 init: %4) across dimensions = [3] : (tensor<2x2x1x2048xf32>, tensor<f32>) -> tensor<2x2x1xf32>
     reducer(%arg44: tensor<f32>, %arg45: tensor<f32>)  {
      %379 = stablehlo.maximum %arg44, %arg45 : tensor<f32>
      stablehlo.return %379 : tensor<f32>
    }
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2] : (tensor<2x2x1xf32>) -> tensor<2x2x1x2048xf32>
    %77 = stablehlo.subtract %74, %76 : tensor<2x2x1x2048xf32>
    %78 = stablehlo.exponential %77 : tensor<2x2x1x2048xf32>
    %79 = stablehlo.reduce(%78 init: %5) across dimensions = [3] : (tensor<2x2x1x2048xf32>, tensor<f32>) -> tensor<2x2x1xf32>
     reducer(%arg44: tensor<f32>, %arg45: tensor<f32>)  {
      %379 = stablehlo.add %arg44, %arg45 : tensor<f32>
      stablehlo.return %379 : tensor<f32>
    }
    %80 = stablehlo.broadcast_in_dim %79, dims = [0, 1, 2] : (tensor<2x2x1xf32>) -> tensor<2x2x1x2048xf32>
    %81 = stablehlo.divide %78, %80 : tensor<2x2x1x2048xf32>
    %82 = stablehlo.reshape %81 : (tensor<2x2x1x2048xf32>) -> tensor<4x1x2048xf32>
    %83 = stablehlo.compare  LT, %arg25, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %84 = stablehlo.reshape %arg26 : (tensor<i64>) -> tensor<1xi64>
    %85 = stablehlo.add %arg25, %84 : tensor<1xi64>
    %86 = stablehlo.select %83, %85, %arg25 : tensor<1xi1>, tensor<1xi64>
    %87 = stablehlo.reshape %86 : (tensor<1xi64>) -> tensor<1x1xi64>
    %88 = stablehlo.reshape %24 : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %89 = stablehlo.transpose %arg20, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %90 = stablehlo.dot %88, %89, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %91 = stablehlo.reshape %90 : (tensor<2x128xf32>) -> tensor<2x1x2x64xf32>
    %92 = "stablehlo.scatter"(%arg27, %87, %91) ({
    ^bb0(%arg44: tensor<f32>, %arg45: tensor<f32>):
      stablehlo.return %arg45 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x2x64xf32>, tensor<1x1xi64>, tensor<2x1x2x64xf32>) -> tensor<2x2304x2x64xf32>
    %93 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %94 = "stablehlo.gather"(%92, %93) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 2, 64]> : tensor<4xi64>} : (tensor<2x2304x2x64xf32>, tensor<2048xui32>) -> tensor<2x2048x2x64xf32>
    %95 = stablehlo.transpose %94, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,2,2048,64]{3,1,2,0}"} : (tensor<2x2048x2x64xf32>) -> tensor<2x2x2048x64xf32>
    %96 = stablehlo.reshape %95 : (tensor<2x2x2048x64xf32>) -> tensor<4x2048x64xf32>
    %97 = stablehlo.dot_general %82, %96, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x1x2048xf32>, tensor<4x2048x64xf32>) -> tensor<4x1x64xf32>
    %98 = stablehlo.reshape %97 : (tensor<4x1x64xf32>) -> tensor<2x128xf32>
    %99 = stablehlo.transpose %arg19, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %100 = stablehlo.dot %98, %99, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %101 = stablehlo.reshape %100 : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
    %102 = stablehlo.add %12, %101 : tensor<2x1x128xf32>
    %103 = stablehlo.power %102, %1 : tensor<2x1x128xf32>
    %104 = stablehlo.reduce(%103 init: %5) across dimensions = [2] : (tensor<2x1x128xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg44: tensor<f32>, %arg45: tensor<f32>)  {
      %379 = stablehlo.add %arg44, %arg45 : tensor<f32>
      stablehlo.return %379 : tensor<f32>
    }
    %105 = stablehlo.multiply %104, %0 : tensor<2x1xf32>
    %106 = stablehlo.reshape %105 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %107 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %108 = stablehlo.add %106, %107 : tensor<2x1x1xf32>
    %109 = stablehlo.rsqrt %108 : tensor<2x1x1xf32>
    %110 = stablehlo.reshape %109 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %111 = stablehlo.broadcast_in_dim %110, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x128xf32>
    %112 = stablehlo.multiply %102, %111 : tensor<2x1x128xf32>
    %113 = stablehlo.broadcast_in_dim %arg18, dims = [2] : (tensor<128xf32>) -> tensor<2x1x128xf32>
    %114 = stablehlo.multiply %112, %113 : tensor<2x1x128xf32>
    %115 = stablehlo.reshape %114 : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %116 = stablehlo.transpose %arg33, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,352]{0,1}"} : (tensor<352x128xf32>) -> tensor<128x352xf32>
    %117 = stablehlo.dot %115, %116, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x352xf32>) -> tensor<2x352xf32>
    %118 = stablehlo.reshape %117 : (tensor<2x352xf32>) -> tensor<2x1x352xf32>
    %119 = stablehlo.logistic %118 : tensor<2x1x352xf32>
    %120 = stablehlo.multiply %118, %119 : tensor<2x1x352xf32>
    %121 = stablehlo.reshape %114 : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %122 = stablehlo.transpose %arg17, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,352]{0,1}"} : (tensor<352x128xf32>) -> tensor<128x352xf32>
    %123 = stablehlo.dot %121, %122, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x352xf32>) -> tensor<2x352xf32>
    %124 = stablehlo.reshape %123 : (tensor<2x352xf32>) -> tensor<2x1x352xf32>
    %125 = stablehlo.multiply %120, %124 : tensor<2x1x352xf32>
    %126 = stablehlo.reshape %125 : (tensor<2x1x352xf32>) -> tensor<2x352xf32>
    %127 = stablehlo.transpose %arg16, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[352,128]{0,1}"} : (tensor<128x352xf32>) -> tensor<352x128xf32>
    %128 = stablehlo.dot %126, %127, precision = [DEFAULT, DEFAULT] : (tensor<2x352xf32>, tensor<352x128xf32>) -> tensor<2x128xf32>
    %129 = stablehlo.reshape %128 : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
    %130 = stablehlo.add %102, %129 : tensor<2x1x128xf32>
    %131 = stablehlo.power %130, %1 : tensor<2x1x128xf32>
    %132 = stablehlo.reduce(%131 init: %5) across dimensions = [2] : (tensor<2x1x128xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg44: tensor<f32>, %arg45: tensor<f32>)  {
      %379 = stablehlo.add %arg44, %arg45 : tensor<f32>
      stablehlo.return %379 : tensor<f32>
    }
    %133 = stablehlo.multiply %132, %0 : tensor<2x1xf32>
    %134 = stablehlo.reshape %133 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %135 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %136 = stablehlo.add %134, %135 : tensor<2x1x1xf32>
    %137 = stablehlo.rsqrt %136 : tensor<2x1x1xf32>
    %138 = stablehlo.reshape %137 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %139 = stablehlo.broadcast_in_dim %138, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x128xf32>
    %140 = stablehlo.multiply %130, %139 : tensor<2x1x128xf32>
    %141 = stablehlo.broadcast_in_dim %arg15, dims = [2] : (tensor<128xf32>) -> tensor<2x1x128xf32>
    %142 = stablehlo.multiply %140, %141 : tensor<2x1x128xf32>
    %143 = stablehlo.reshape %142 : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %144 = stablehlo.transpose %arg37, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %145 = stablehlo.dot %143, %144, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %146 = stablehlo.reshape %145 : (tensor<2x128xf32>) -> tensor<2x1x2x32x2xf32>
    %147 = stablehlo.slice %146 [0:2, 0:1, 0:2, 0:32, 0:1] : (tensor<2x1x2x32x2xf32>) -> tensor<2x1x2x32x1xf32>
    %148 = stablehlo.reshape %147 : (tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32xf32>
    %149 = stablehlo.slice %146 [0:2, 0:1, 0:2, 0:32, 1:2] : (tensor<2x1x2x32x2xf32>) -> tensor<2x1x2x32x1xf32>
    %150 = stablehlo.reshape %149 : (tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32xf32>
    %151 = stablehlo.complex %148, %150 : tensor<2x1x2x32xcomplex<f32>>
    %152 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x32xcomplex<f32>>) -> tensor<2x1x2x32xcomplex<f32>>
    %153 = stablehlo.multiply %151, %152 : tensor<2x1x2x32xcomplex<f32>>
    %154 = stablehlo.real %153 : (tensor<2x1x2x32xcomplex<f32>>) -> tensor<2x1x2x32xf32>
    %155 = stablehlo.reshape %154 : (tensor<2x1x2x32xf32>) -> tensor<2x1x2x32x1xf32>
    %156 = stablehlo.imag %153 : (tensor<2x1x2x32xcomplex<f32>>) -> tensor<2x1x2x32xf32>
    %157 = stablehlo.reshape %156 : (tensor<2x1x2x32xf32>) -> tensor<2x1x2x32x1xf32>
    %158 = stablehlo.concatenate %155, %157, dim = 4 : (tensor<2x1x2x32x1xf32>, tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32x2xf32>
    %159 = stablehlo.reshape %158 : (tensor<2x1x2x32x2xf32>) -> tensor<4x1x64xf32>
    %160 = stablehlo.compare  LT, %arg25, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %161 = stablehlo.reshape %arg26 : (tensor<i64>) -> tensor<1xi64>
    %162 = stablehlo.add %arg25, %161 : tensor<1xi64>
    %163 = stablehlo.select %160, %162, %arg25 : tensor<1xi1>, tensor<1xi64>
    %164 = stablehlo.reshape %163 : (tensor<1xi64>) -> tensor<1x1xi64>
    %165 = stablehlo.reshape %142 : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %166 = stablehlo.transpose %arg35, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %167 = stablehlo.dot %165, %166, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %168 = stablehlo.reshape %167 : (tensor<2x128xf32>) -> tensor<2x1x2x32x2xf32>
    %169 = stablehlo.slice %168 [0:2, 0:1, 0:2, 0:32, 0:1] : (tensor<2x1x2x32x2xf32>) -> tensor<2x1x2x32x1xf32>
    %170 = stablehlo.reshape %169 : (tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32xf32>
    %171 = stablehlo.slice %168 [0:2, 0:1, 0:2, 0:32, 1:2] : (tensor<2x1x2x32x2xf32>) -> tensor<2x1x2x32x1xf32>
    %172 = stablehlo.reshape %171 : (tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32xf32>
    %173 = stablehlo.complex %170, %172 : tensor<2x1x2x32xcomplex<f32>>
    %174 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x32xcomplex<f32>>) -> tensor<2x1x2x32xcomplex<f32>>
    %175 = stablehlo.multiply %173, %174 : tensor<2x1x2x32xcomplex<f32>>
    %176 = stablehlo.real %175 : (tensor<2x1x2x32xcomplex<f32>>) -> tensor<2x1x2x32xf32>
    %177 = stablehlo.reshape %176 : (tensor<2x1x2x32xf32>) -> tensor<2x1x2x32x1xf32>
    %178 = stablehlo.imag %175 : (tensor<2x1x2x32xcomplex<f32>>) -> tensor<2x1x2x32xf32>
    %179 = stablehlo.reshape %178 : (tensor<2x1x2x32xf32>) -> tensor<2x1x2x32x1xf32>
    %180 = stablehlo.concatenate %177, %179, dim = 4 : (tensor<2x1x2x32x1xf32>, tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32x2xf32>
    %181 = stablehlo.reshape %180 : (tensor<2x1x2x32x2xf32>) -> tensor<2x1x2x64xf32>
    %182 = "stablehlo.scatter"(%arg36, %164, %181) ({
    ^bb0(%arg44: tensor<f32>, %arg45: tensor<f32>):
      stablehlo.return %arg45 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x2x64xf32>, tensor<1x1xi64>, tensor<2x1x2x64xf32>) -> tensor<2x2304x2x64xf32>
    %183 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %184 = "stablehlo.gather"(%182, %183) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 2, 64]> : tensor<4xi64>} : (tensor<2x2304x2x64xf32>, tensor<2048xui32>) -> tensor<2x2048x2x64xf32>
    %185 = stablehlo.transpose %184, dims = [0, 2, 3, 1] : (tensor<2x2048x2x64xf32>) -> tensor<2x2x64x2048xf32>
    %186 = stablehlo.reshape %185 : (tensor<2x2x64x2048xf32>) -> tensor<4x64x2048xf32>
    %187 = stablehlo.dot_general %159, %186, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x1x64xf32>, tensor<4x64x2048xf32>) -> tensor<4x1x2048xf32>
    %188 = stablehlo.reshape %187 : (tensor<4x1x2048xf32>) -> tensor<2x2x1x2048xf32>
    %189 = stablehlo.broadcast_in_dim %arg28, dims = [] : (tensor<f32>) -> tensor<2x2x1x2048xf32>
    %190 = stablehlo.divide %188, %189 : tensor<2x2x1x2048xf32>
    %191 = stablehlo.reduce(%190 init: %4) across dimensions = [3] : (tensor<2x2x1x2048xf32>, tensor<f32>) -> tensor<2x2x1xf32>
     reducer(%arg44: tensor<f32>, %arg45: tensor<f32>)  {
      %379 = stablehlo.maximum %arg44, %arg45 : tensor<f32>
      stablehlo.return %379 : tensor<f32>
    }
    %192 = stablehlo.broadcast_in_dim %191, dims = [0, 1, 2] : (tensor<2x2x1xf32>) -> tensor<2x2x1x2048xf32>
    %193 = stablehlo.subtract %190, %192 : tensor<2x2x1x2048xf32>
    %194 = stablehlo.exponential %193 : tensor<2x2x1x2048xf32>
    %195 = stablehlo.reduce(%194 init: %5) across dimensions = [3] : (tensor<2x2x1x2048xf32>, tensor<f32>) -> tensor<2x2x1xf32>
     reducer(%arg44: tensor<f32>, %arg45: tensor<f32>)  {
      %379 = stablehlo.add %arg44, %arg45 : tensor<f32>
      stablehlo.return %379 : tensor<f32>
    }
    %196 = stablehlo.broadcast_in_dim %195, dims = [0, 1, 2] : (tensor<2x2x1xf32>) -> tensor<2x2x1x2048xf32>
    %197 = stablehlo.divide %194, %196 : tensor<2x2x1x2048xf32>
    %198 = stablehlo.reshape %197 : (tensor<2x2x1x2048xf32>) -> tensor<4x1x2048xf32>
    %199 = stablehlo.compare  LT, %arg25, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %200 = stablehlo.reshape %arg26 : (tensor<i64>) -> tensor<1xi64>
    %201 = stablehlo.add %arg25, %200 : tensor<1xi64>
    %202 = stablehlo.select %199, %201, %arg25 : tensor<1xi1>, tensor<1xi64>
    %203 = stablehlo.reshape %202 : (tensor<1xi64>) -> tensor<1x1xi64>
    %204 = stablehlo.reshape %142 : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %205 = stablehlo.transpose %arg14, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %206 = stablehlo.dot %204, %205, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %207 = stablehlo.reshape %206 : (tensor<2x128xf32>) -> tensor<2x1x2x64xf32>
    %208 = "stablehlo.scatter"(%arg34, %203, %207) ({
    ^bb0(%arg44: tensor<f32>, %arg45: tensor<f32>):
      stablehlo.return %arg45 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x2x64xf32>, tensor<1x1xi64>, tensor<2x1x2x64xf32>) -> tensor<2x2304x2x64xf32>
    %209 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %210 = "stablehlo.gather"(%208, %209) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 2, 64]> : tensor<4xi64>} : (tensor<2x2304x2x64xf32>, tensor<2048xui32>) -> tensor<2x2048x2x64xf32>
    %211 = stablehlo.transpose %210, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,2,2048,64]{3,1,2,0}"} : (tensor<2x2048x2x64xf32>) -> tensor<2x2x2048x64xf32>
    %212 = stablehlo.reshape %211 : (tensor<2x2x2048x64xf32>) -> tensor<4x2048x64xf32>
    %213 = stablehlo.dot_general %198, %212, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x1x2048xf32>, tensor<4x2048x64xf32>) -> tensor<4x1x64xf32>
    %214 = stablehlo.reshape %213 : (tensor<4x1x64xf32>) -> tensor<2x128xf32>
    %215 = stablehlo.transpose %arg13, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %216 = stablehlo.dot %214, %215, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %217 = stablehlo.reshape %216 : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
    %218 = stablehlo.add %130, %217 : tensor<2x1x128xf32>
    %219 = stablehlo.power %218, %1 : tensor<2x1x128xf32>
    %220 = stablehlo.reduce(%219 init: %5) across dimensions = [2] : (tensor<2x1x128xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg44: tensor<f32>, %arg45: tensor<f32>)  {
      %379 = stablehlo.add %arg44, %arg45 : tensor<f32>
      stablehlo.return %379 : tensor<f32>
    }
    %221 = stablehlo.multiply %220, %0 : tensor<2x1xf32>
    %222 = stablehlo.reshape %221 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %223 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %224 = stablehlo.add %222, %223 : tensor<2x1x1xf32>
    %225 = stablehlo.rsqrt %224 : tensor<2x1x1xf32>
    %226 = stablehlo.reshape %225 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %227 = stablehlo.broadcast_in_dim %226, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x128xf32>
    %228 = stablehlo.multiply %218, %227 : tensor<2x1x128xf32>
    %229 = stablehlo.broadcast_in_dim %arg12, dims = [2] : (tensor<128xf32>) -> tensor<2x1x128xf32>
    %230 = stablehlo.multiply %228, %229 : tensor<2x1x128xf32>
    %231 = stablehlo.reshape %230 : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %232 = stablehlo.transpose %arg38, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,352]{0,1}"} : (tensor<352x128xf32>) -> tensor<128x352xf32>
    %233 = stablehlo.dot %231, %232, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x352xf32>) -> tensor<2x352xf32>
    %234 = stablehlo.reshape %233 : (tensor<2x352xf32>) -> tensor<2x1x352xf32>
    %235 = stablehlo.logistic %234 : tensor<2x1x352xf32>
    %236 = stablehlo.multiply %234, %235 : tensor<2x1x352xf32>
    %237 = stablehlo.reshape %230 : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %238 = stablehlo.transpose %arg11, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,352]{0,1}"} : (tensor<352x128xf32>) -> tensor<128x352xf32>
    %239 = stablehlo.dot %237, %238, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x352xf32>) -> tensor<2x352xf32>
    %240 = stablehlo.reshape %239 : (tensor<2x352xf32>) -> tensor<2x1x352xf32>
    %241 = stablehlo.multiply %236, %240 : tensor<2x1x352xf32>
    %242 = stablehlo.reshape %241 : (tensor<2x1x352xf32>) -> tensor<2x352xf32>
    %243 = stablehlo.transpose %arg10, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[352,128]{0,1}"} : (tensor<128x352xf32>) -> tensor<352x128xf32>
    %244 = stablehlo.dot %242, %243, precision = [DEFAULT, DEFAULT] : (tensor<2x352xf32>, tensor<352x128xf32>) -> tensor<2x128xf32>
    %245 = stablehlo.reshape %244 : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
    %246 = stablehlo.add %218, %245 : tensor<2x1x128xf32>
    %247 = stablehlo.power %246, %1 : tensor<2x1x128xf32>
    %248 = stablehlo.reduce(%247 init: %5) across dimensions = [2] : (tensor<2x1x128xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg44: tensor<f32>, %arg45: tensor<f32>)  {
      %379 = stablehlo.add %arg44, %arg45 : tensor<f32>
      stablehlo.return %379 : tensor<f32>
    }
    %249 = stablehlo.multiply %248, %0 : tensor<2x1xf32>
    %250 = stablehlo.reshape %249 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %251 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %252 = stablehlo.add %250, %251 : tensor<2x1x1xf32>
    %253 = stablehlo.rsqrt %252 : tensor<2x1x1xf32>
    %254 = stablehlo.reshape %253 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %255 = stablehlo.broadcast_in_dim %254, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x128xf32>
    %256 = stablehlo.multiply %246, %255 : tensor<2x1x128xf32>
    %257 = stablehlo.broadcast_in_dim %arg9, dims = [2] : (tensor<128xf32>) -> tensor<2x1x128xf32>
    %258 = stablehlo.multiply %256, %257 : tensor<2x1x128xf32>
    %259 = stablehlo.reshape %258 : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %260 = stablehlo.transpose %arg42, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %261 = stablehlo.dot %259, %260, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %262 = stablehlo.reshape %261 : (tensor<2x128xf32>) -> tensor<2x1x2x32x2xf32>
    %263 = stablehlo.slice %262 [0:2, 0:1, 0:2, 0:32, 0:1] : (tensor<2x1x2x32x2xf32>) -> tensor<2x1x2x32x1xf32>
    %264 = stablehlo.reshape %263 : (tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32xf32>
    %265 = stablehlo.slice %262 [0:2, 0:1, 0:2, 0:32, 1:2] : (tensor<2x1x2x32x2xf32>) -> tensor<2x1x2x32x1xf32>
    %266 = stablehlo.reshape %265 : (tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32xf32>
    %267 = stablehlo.complex %264, %266 : tensor<2x1x2x32xcomplex<f32>>
    %268 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x32xcomplex<f32>>) -> tensor<2x1x2x32xcomplex<f32>>
    %269 = stablehlo.multiply %267, %268 : tensor<2x1x2x32xcomplex<f32>>
    %270 = stablehlo.real %269 : (tensor<2x1x2x32xcomplex<f32>>) -> tensor<2x1x2x32xf32>
    %271 = stablehlo.reshape %270 : (tensor<2x1x2x32xf32>) -> tensor<2x1x2x32x1xf32>
    %272 = stablehlo.imag %269 : (tensor<2x1x2x32xcomplex<f32>>) -> tensor<2x1x2x32xf32>
    %273 = stablehlo.reshape %272 : (tensor<2x1x2x32xf32>) -> tensor<2x1x2x32x1xf32>
    %274 = stablehlo.concatenate %271, %273, dim = 4 : (tensor<2x1x2x32x1xf32>, tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32x2xf32>
    %275 = stablehlo.reshape %274 : (tensor<2x1x2x32x2xf32>) -> tensor<4x1x64xf32>
    %276 = stablehlo.compare  LT, %arg25, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %277 = stablehlo.reshape %arg26 : (tensor<i64>) -> tensor<1xi64>
    %278 = stablehlo.add %arg25, %277 : tensor<1xi64>
    %279 = stablehlo.select %276, %278, %arg25 : tensor<1xi1>, tensor<1xi64>
    %280 = stablehlo.reshape %279 : (tensor<1xi64>) -> tensor<1x1xi64>
    %281 = stablehlo.reshape %258 : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %282 = stablehlo.transpose %arg40, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %283 = stablehlo.dot %281, %282, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %284 = stablehlo.reshape %283 : (tensor<2x128xf32>) -> tensor<2x1x2x32x2xf32>
    %285 = stablehlo.slice %284 [0:2, 0:1, 0:2, 0:32, 0:1] : (tensor<2x1x2x32x2xf32>) -> tensor<2x1x2x32x1xf32>
    %286 = stablehlo.reshape %285 : (tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32xf32>
    %287 = stablehlo.slice %284 [0:2, 0:1, 0:2, 0:32, 1:2] : (tensor<2x1x2x32x2xf32>) -> tensor<2x1x2x32x1xf32>
    %288 = stablehlo.reshape %287 : (tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32xf32>
    %289 = stablehlo.complex %286, %288 : tensor<2x1x2x32xcomplex<f32>>
    %290 = stablehlo.broadcast_in_dim %35, dims = [1, 3] : (tensor<1x32xcomplex<f32>>) -> tensor<2x1x2x32xcomplex<f32>>
    %291 = stablehlo.multiply %289, %290 : tensor<2x1x2x32xcomplex<f32>>
    %292 = stablehlo.real %291 : (tensor<2x1x2x32xcomplex<f32>>) -> tensor<2x1x2x32xf32>
    %293 = stablehlo.reshape %292 : (tensor<2x1x2x32xf32>) -> tensor<2x1x2x32x1xf32>
    %294 = stablehlo.imag %291 : (tensor<2x1x2x32xcomplex<f32>>) -> tensor<2x1x2x32xf32>
    %295 = stablehlo.reshape %294 : (tensor<2x1x2x32xf32>) -> tensor<2x1x2x32x1xf32>
    %296 = stablehlo.concatenate %293, %295, dim = 4 : (tensor<2x1x2x32x1xf32>, tensor<2x1x2x32x1xf32>) -> tensor<2x1x2x32x2xf32>
    %297 = stablehlo.reshape %296 : (tensor<2x1x2x32x2xf32>) -> tensor<2x1x2x64xf32>
    %298 = "stablehlo.scatter"(%arg41, %280, %297) ({
    ^bb0(%arg44: tensor<f32>, %arg45: tensor<f32>):
      stablehlo.return %arg45 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x2x64xf32>, tensor<1x1xi64>, tensor<2x1x2x64xf32>) -> tensor<2x2304x2x64xf32>
    %299 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %300 = "stablehlo.gather"(%298, %299) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 2, 64]> : tensor<4xi64>} : (tensor<2x2304x2x64xf32>, tensor<2048xui32>) -> tensor<2x2048x2x64xf32>
    %301 = stablehlo.transpose %300, dims = [0, 2, 3, 1] : (tensor<2x2048x2x64xf32>) -> tensor<2x2x64x2048xf32>
    %302 = stablehlo.reshape %301 : (tensor<2x2x64x2048xf32>) -> tensor<4x64x2048xf32>
    %303 = stablehlo.dot_general %275, %302, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x1x64xf32>, tensor<4x64x2048xf32>) -> tensor<4x1x2048xf32>
    %304 = stablehlo.reshape %303 : (tensor<4x1x2048xf32>) -> tensor<2x2x1x2048xf32>
    %305 = stablehlo.broadcast_in_dim %arg28, dims = [] : (tensor<f32>) -> tensor<2x2x1x2048xf32>
    %306 = stablehlo.divide %304, %305 : tensor<2x2x1x2048xf32>
    %307 = stablehlo.reduce(%306 init: %4) across dimensions = [3] : (tensor<2x2x1x2048xf32>, tensor<f32>) -> tensor<2x2x1xf32>
     reducer(%arg44: tensor<f32>, %arg45: tensor<f32>)  {
      %379 = stablehlo.maximum %arg44, %arg45 : tensor<f32>
      stablehlo.return %379 : tensor<f32>
    }
    %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1, 2] : (tensor<2x2x1xf32>) -> tensor<2x2x1x2048xf32>
    %309 = stablehlo.subtract %306, %308 : tensor<2x2x1x2048xf32>
    %310 = stablehlo.exponential %309 : tensor<2x2x1x2048xf32>
    %311 = stablehlo.reduce(%310 init: %5) across dimensions = [3] : (tensor<2x2x1x2048xf32>, tensor<f32>) -> tensor<2x2x1xf32>
     reducer(%arg44: tensor<f32>, %arg45: tensor<f32>)  {
      %379 = stablehlo.add %arg44, %arg45 : tensor<f32>
      stablehlo.return %379 : tensor<f32>
    }
    %312 = stablehlo.broadcast_in_dim %311, dims = [0, 1, 2] : (tensor<2x2x1xf32>) -> tensor<2x2x1x2048xf32>
    %313 = stablehlo.divide %310, %312 : tensor<2x2x1x2048xf32>
    %314 = stablehlo.reshape %313 : (tensor<2x2x1x2048xf32>) -> tensor<4x1x2048xf32>
    %315 = stablehlo.compare  LT, %arg25, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %316 = stablehlo.reshape %arg26 : (tensor<i64>) -> tensor<1xi64>
    %317 = stablehlo.add %arg25, %316 : tensor<1xi64>
    %318 = stablehlo.select %315, %317, %arg25 : tensor<1xi1>, tensor<1xi64>
    %319 = stablehlo.reshape %318 : (tensor<1xi64>) -> tensor<1x1xi64>
    %320 = stablehlo.reshape %258 : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %321 = stablehlo.transpose %arg8, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %322 = stablehlo.dot %320, %321, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %323 = stablehlo.reshape %322 : (tensor<2x128xf32>) -> tensor<2x1x2x64xf32>
    %324 = "stablehlo.scatter"(%arg39, %319, %323) ({
    ^bb0(%arg44: tensor<f32>, %arg45: tensor<f32>):
      stablehlo.return %arg45 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2304x2x64xf32>, tensor<1x1xi64>, tensor<2x1x2x64xf32>) -> tensor<2x2304x2x64xf32>
    %325 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %326 = "stablehlo.gather"(%324, %325) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[2, 1, 2, 64]> : tensor<4xi64>} : (tensor<2x2304x2x64xf32>, tensor<2048xui32>) -> tensor<2x2048x2x64xf32>
    %327 = stablehlo.transpose %326, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[2,2,2048,64]{3,1,2,0}"} : (tensor<2x2048x2x64xf32>) -> tensor<2x2x2048x64xf32>
    %328 = stablehlo.reshape %327 : (tensor<2x2x2048x64xf32>) -> tensor<4x2048x64xf32>
    %329 = stablehlo.dot_general %314, %328, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x1x2048xf32>, tensor<4x2048x64xf32>) -> tensor<4x1x64xf32>
    %330 = stablehlo.reshape %329 : (tensor<4x1x64xf32>) -> tensor<2x128xf32>
    %331 = stablehlo.transpose %arg6, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %332 = stablehlo.dot %330, %331, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x128xf32>) -> tensor<2x128xf32>
    %333 = stablehlo.reshape %332 : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
    %334 = stablehlo.add %246, %333 : tensor<2x1x128xf32>
    %335 = stablehlo.power %334, %1 : tensor<2x1x128xf32>
    %336 = stablehlo.reduce(%335 init: %5) across dimensions = [2] : (tensor<2x1x128xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg44: tensor<f32>, %arg45: tensor<f32>)  {
      %379 = stablehlo.add %arg44, %arg45 : tensor<f32>
      stablehlo.return %379 : tensor<f32>
    }
    %337 = stablehlo.multiply %336, %0 : tensor<2x1xf32>
    %338 = stablehlo.reshape %337 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %339 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %340 = stablehlo.add %338, %339 : tensor<2x1x1xf32>
    %341 = stablehlo.rsqrt %340 : tensor<2x1x1xf32>
    %342 = stablehlo.reshape %341 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %343 = stablehlo.broadcast_in_dim %342, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x128xf32>
    %344 = stablehlo.multiply %334, %343 : tensor<2x1x128xf32>
    %345 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<128xf32>) -> tensor<2x1x128xf32>
    %346 = stablehlo.multiply %344, %345 : tensor<2x1x128xf32>
    %347 = stablehlo.reshape %346 : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %348 = stablehlo.transpose %arg43, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,352]{0,1}"} : (tensor<352x128xf32>) -> tensor<128x352xf32>
    %349 = stablehlo.dot %347, %348, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x352xf32>) -> tensor<2x352xf32>
    %350 = stablehlo.reshape %349 : (tensor<2x352xf32>) -> tensor<2x1x352xf32>
    %351 = stablehlo.logistic %350 : tensor<2x1x352xf32>
    %352 = stablehlo.multiply %350, %351 : tensor<2x1x352xf32>
    %353 = stablehlo.reshape %346 : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %354 = stablehlo.transpose %arg4, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,352]{0,1}"} : (tensor<352x128xf32>) -> tensor<128x352xf32>
    %355 = stablehlo.dot %353, %354, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x352xf32>) -> tensor<2x352xf32>
    %356 = stablehlo.reshape %355 : (tensor<2x352xf32>) -> tensor<2x1x352xf32>
    %357 = stablehlo.multiply %352, %356 : tensor<2x1x352xf32>
    %358 = stablehlo.reshape %357 : (tensor<2x1x352xf32>) -> tensor<2x352xf32>
    %359 = stablehlo.transpose %arg3, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[352,128]{0,1}"} : (tensor<128x352xf32>) -> tensor<352x128xf32>
    %360 = stablehlo.dot %358, %359, precision = [DEFAULT, DEFAULT] : (tensor<2x352xf32>, tensor<352x128xf32>) -> tensor<2x128xf32>
    %361 = stablehlo.reshape %360 : (tensor<2x128xf32>) -> tensor<2x1x128xf32>
    %362 = stablehlo.add %334, %361 : tensor<2x1x128xf32>
    %363 = stablehlo.power %362, %1 : tensor<2x1x128xf32>
    %364 = stablehlo.reduce(%363 init: %5) across dimensions = [2] : (tensor<2x1x128xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg44: tensor<f32>, %arg45: tensor<f32>)  {
      %379 = stablehlo.add %arg44, %arg45 : tensor<f32>
      stablehlo.return %379 : tensor<f32>
    }
    %365 = stablehlo.multiply %364, %0 : tensor<2x1xf32>
    %366 = stablehlo.reshape %365 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
    %367 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<2x1x1xf32>
    %368 = stablehlo.add %366, %367 : tensor<2x1x1xf32>
    %369 = stablehlo.rsqrt %368 : tensor<2x1x1xf32>
    %370 = stablehlo.reshape %369 : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
    %371 = stablehlo.broadcast_in_dim %370, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x128xf32>
    %372 = stablehlo.multiply %362, %371 : tensor<2x1x128xf32>
    %373 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<128xf32>) -> tensor<2x1x128xf32>
    %374 = stablehlo.multiply %372, %373 : tensor<2x1x128xf32>
    %375 = stablehlo.reshape %374 : (tensor<2x1x128xf32>) -> tensor<2x128xf32>
    %376 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,32000]{0,1}"} : (tensor<32000x128xf32>) -> tensor<128x32000xf32>
    %377 = stablehlo.dot %375, %376, precision = [DEFAULT, DEFAULT] : (tensor<2x128xf32>, tensor<128x32000xf32>) -> tensor<2x32000xf32>
    %378 = stablehlo.reshape %377 : (tensor<2x32000xf32>) -> tensor<2x1x32000xf32>
    return %378, %66, %92, %182, %208, %298, %324 : tensor<2x1x32000xf32>, tensor<2x2304x2x64xf32>, tensor<2x2304x2x64xf32>, tensor<2x2304x2x64xf32>, tensor<2x2304x2x64xf32>, tensor<2x2304x2x64xf32>, tensor<2x2304x2x64xf32>
  }
}
