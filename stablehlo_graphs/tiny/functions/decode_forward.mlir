module @IrToHlo.692 attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<32000x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<f32>, %arg3: tensor<128x352xf32>, %arg4: tensor<352x128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<128x128xf32>, %arg7: tensor<2048xi64>, %arg8: tensor<128x128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<128x352xf32>, %arg11: tensor<352x128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<128x128xf32>, %arg14: tensor<128x128xf32>, %arg15: tensor<128xf32>, %arg16: tensor<128x352xf32>, %arg17: tensor<352x128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<128x128xf32>, %arg20: tensor<128x128xf32>, %arg21: tensor<128xf32>, %arg22: tensor<1xi64>, %arg23: tensor<32000x128xf32>, %arg24: tensor<1xi64>, %arg25: tensor<i64>, %arg26: tensor<2304x2x64xf32>, %arg27: tensor<f32>, %arg28: tensor<4608x32xcomplex<f32>>, %arg29: tensor<128x128xf32>, %arg30: tensor<2304x2x64xf32>, %arg31: tensor<128x128xf32>, %arg32: tensor<352x128xf32>, %arg33: tensor<2304x2x64xf32>, %arg34: tensor<128x128xf32>, %arg35: tensor<2304x2x64xf32>, %arg36: tensor<128x128xf32>, %arg37: tensor<352x128xf32>, %arg38: tensor<2304x2x64xf32>, %arg39: tensor<128x128xf32>, %arg40: tensor<2304x2x64xf32>, %arg41: tensor<128x128xf32>, %arg42: tensor<352x128xf32>) -> (tensor<1x32000xf32>, tensor<2304x2x64xf32>, tensor<2304x2x64xf32>, tensor<2304x2x64xf32>, tensor<2304x2x64xf32>, tensor<2304x2x64xf32>, tensor<2304x2x64xf32>) {
    %0 = stablehlo.constant dense<7.812500e-03> : tensor<1xf32>
    %1 = stablehlo.constant dense<2.000000e+00> : tensor<1x128xf32>
    %2 = stablehlo.constant dense<0> : tensor<1xi64>
    %3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.convert %arg22 : (tensor<1xi64>) -> tensor<1xui32>
    %6 = "stablehlo.gather"(%arg23, %5) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<32000x128xf32>, tensor<1xui32>) -> tensor<1x128xf32>
    %7 = stablehlo.power %6, %1 : tensor<1x128xf32>
    %8 = stablehlo.reduce(%7 init: %4) across dimensions = [1] : (tensor<1x128xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg43: tensor<f32>, %arg44: tensor<f32>)  {
      %329 = stablehlo.add %arg43, %arg44 : tensor<f32>
      stablehlo.return %329 : tensor<f32>
    }
    %9 = stablehlo.multiply %8, %0 : tensor<1xf32>
    %10 = stablehlo.reshape %9 : (tensor<1xf32>) -> tensor<1x1xf32>
    %11 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %12 = stablehlo.add %10, %11 : tensor<1x1xf32>
    %13 = stablehlo.rsqrt %12 : tensor<1x1xf32>
    %14 = stablehlo.reshape %13 : (tensor<1x1xf32>) -> tensor<1xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<1xf32>) -> tensor<1x128xf32>
    %16 = stablehlo.multiply %6, %15 : tensor<1x128xf32>
    %17 = stablehlo.reshape %arg21 : (tensor<128xf32>) -> tensor<1x128xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<1x128xf32>
    %19 = stablehlo.transpose %arg31, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %20 = stablehlo.dot %18, %19, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %21 = stablehlo.reshape %20 : (tensor<1x128xf32>) -> tensor<1x2x32x2xf32>
    %22 = stablehlo.slice %21 [0:1, 0:2, 0:32, 0:1] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x1xf32>
    %23 = stablehlo.reshape %22 : (tensor<1x2x32x1xf32>) -> tensor<1x2x32xf32>
    %24 = stablehlo.slice %21 [0:1, 0:2, 0:32, 1:2] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x1xf32>
    %25 = stablehlo.reshape %24 : (tensor<1x2x32x1xf32>) -> tensor<1x2x32xf32>
    %26 = stablehlo.complex %23, %25 : tensor<1x2x32xcomplex<f32>>
    %27 = stablehlo.convert %arg24 : (tensor<1xi64>) -> tensor<1xui32>
    %28 = "stablehlo.gather"(%arg28, %27) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 32]> : tensor<2xi64>} : (tensor<4608x32xcomplex<f32>>, tensor<1xui32>) -> tensor<1x32xcomplex<f32>>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x32xcomplex<f32>>) -> tensor<1x2x32xcomplex<f32>>
    %30 = stablehlo.multiply %26, %29 : tensor<1x2x32xcomplex<f32>>
    %31 = stablehlo.real %30 : (tensor<1x2x32xcomplex<f32>>) -> tensor<1x2x32xf32>
    %32 = stablehlo.reshape %31 : (tensor<1x2x32xf32>) -> tensor<1x2x32x1xf32>
    %33 = stablehlo.imag %30 : (tensor<1x2x32xcomplex<f32>>) -> tensor<1x2x32xf32>
    %34 = stablehlo.reshape %33 : (tensor<1x2x32xf32>) -> tensor<1x2x32x1xf32>
    %35 = stablehlo.concatenate %32, %34, dim = 3 : (tensor<1x2x32x1xf32>, tensor<1x2x32x1xf32>) -> tensor<1x2x32x2xf32>
    %36 = stablehlo.reshape %35 : (tensor<1x2x32x2xf32>) -> tensor<2x1x64xf32>
    %37 = stablehlo.compare  LT, %arg24, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %38 = stablehlo.reshape %arg25 : (tensor<i64>) -> tensor<1xi64>
    %39 = stablehlo.add %arg24, %38 : tensor<1xi64>
    %40 = stablehlo.select %37, %39, %arg24 : tensor<1xi1>, tensor<1xi64>
    %41 = stablehlo.reshape %40 : (tensor<1xi64>) -> tensor<1x1xi64>
    %42 = stablehlo.transpose %arg29, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %43 = stablehlo.dot %18, %42, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %44 = stablehlo.reshape %43 : (tensor<1x128xf32>) -> tensor<1x2x32x2xf32>
    %45 = stablehlo.slice %44 [0:1, 0:2, 0:32, 0:1] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x1xf32>
    %46 = stablehlo.reshape %45 : (tensor<1x2x32x1xf32>) -> tensor<1x2x32xf32>
    %47 = stablehlo.slice %44 [0:1, 0:2, 0:32, 1:2] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x1xf32>
    %48 = stablehlo.reshape %47 : (tensor<1x2x32x1xf32>) -> tensor<1x2x32xf32>
    %49 = stablehlo.complex %46, %48 : tensor<1x2x32xcomplex<f32>>
    %50 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x32xcomplex<f32>>) -> tensor<1x2x32xcomplex<f32>>
    %51 = stablehlo.multiply %49, %50 : tensor<1x2x32xcomplex<f32>>
    %52 = stablehlo.real %51 : (tensor<1x2x32xcomplex<f32>>) -> tensor<1x2x32xf32>
    %53 = stablehlo.reshape %52 : (tensor<1x2x32xf32>) -> tensor<1x2x32x1xf32>
    %54 = stablehlo.imag %51 : (tensor<1x2x32xcomplex<f32>>) -> tensor<1x2x32xf32>
    %55 = stablehlo.reshape %54 : (tensor<1x2x32xf32>) -> tensor<1x2x32x1xf32>
    %56 = stablehlo.concatenate %53, %55, dim = 3 : (tensor<1x2x32x1xf32>, tensor<1x2x32x1xf32>) -> tensor<1x2x32x2xf32>
    %57 = stablehlo.reshape %56 : (tensor<1x2x32x2xf32>) -> tensor<1x2x64xf32>
    %58 = "stablehlo.scatter"(%arg30, %41, %57) ({
    ^bb0(%arg43: tensor<f32>, %arg44: tensor<f32>):
      stablehlo.return %arg44 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x2x64xf32>, tensor<1x1xi64>, tensor<1x2x64xf32>) -> tensor<2304x2x64xf32>
    %59 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %60 = "stablehlo.gather"(%58, %59) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 2, 64]> : tensor<3xi64>} : (tensor<2304x2x64xf32>, tensor<2048xui32>) -> tensor<2048x2x64xf32>
    %61 = stablehlo.transpose %60, dims = [1, 2, 0] : (tensor<2048x2x64xf32>) -> tensor<2x64x2048xf32>
    %62 = stablehlo.dot_general %36, %61, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x1x64xf32>, tensor<2x64x2048xf32>) -> tensor<2x1x2048xf32>
    %63 = stablehlo.broadcast_in_dim %arg27, dims = [] : (tensor<f32>) -> tensor<2x1x2048xf32>
    %64 = stablehlo.divide %62, %63 : tensor<2x1x2048xf32>
    %65 = stablehlo.reduce(%64 init: %3) across dimensions = [2] : (tensor<2x1x2048xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg43: tensor<f32>, %arg44: tensor<f32>)  {
      %329 = stablehlo.maximum %arg43, %arg44 : tensor<f32>
      stablehlo.return %329 : tensor<f32>
    }
    %66 = stablehlo.broadcast_in_dim %65, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x2048xf32>
    %67 = stablehlo.subtract %64, %66 : tensor<2x1x2048xf32>
    %68 = stablehlo.exponential %67 : tensor<2x1x2048xf32>
    %69 = stablehlo.reduce(%68 init: %4) across dimensions = [2] : (tensor<2x1x2048xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg43: tensor<f32>, %arg44: tensor<f32>)  {
      %329 = stablehlo.add %arg43, %arg44 : tensor<f32>
      stablehlo.return %329 : tensor<f32>
    }
    %70 = stablehlo.broadcast_in_dim %69, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x2048xf32>
    %71 = stablehlo.divide %68, %70 : tensor<2x1x2048xf32>
    %72 = stablehlo.compare  LT, %arg24, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %73 = stablehlo.reshape %arg25 : (tensor<i64>) -> tensor<1xi64>
    %74 = stablehlo.add %arg24, %73 : tensor<1xi64>
    %75 = stablehlo.select %72, %74, %arg24 : tensor<1xi1>, tensor<1xi64>
    %76 = stablehlo.reshape %75 : (tensor<1xi64>) -> tensor<1x1xi64>
    %77 = stablehlo.transpose %arg20, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %78 = stablehlo.dot %18, %77, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %79 = stablehlo.reshape %78 : (tensor<1x128xf32>) -> tensor<1x2x64xf32>
    %80 = "stablehlo.scatter"(%arg26, %76, %79) ({
    ^bb0(%arg43: tensor<f32>, %arg44: tensor<f32>):
      stablehlo.return %arg44 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x2x64xf32>, tensor<1x1xi64>, tensor<1x2x64xf32>) -> tensor<2304x2x64xf32>
    %81 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %82 = "stablehlo.gather"(%80, %81) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 2, 64]> : tensor<3xi64>} : (tensor<2304x2x64xf32>, tensor<2048xui32>) -> tensor<2048x2x64xf32>
    %83 = stablehlo.transpose %82, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[2,2048,64]{2,0,1}"} : (tensor<2048x2x64xf32>) -> tensor<2x2048x64xf32>
    %84 = stablehlo.dot_general %71, %83, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x1x2048xf32>, tensor<2x2048x64xf32>) -> tensor<2x1x64xf32>
    %85 = stablehlo.reshape %84 : (tensor<2x1x64xf32>) -> tensor<1x128xf32>
    %86 = stablehlo.transpose %arg19, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %87 = stablehlo.dot %85, %86, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %88 = stablehlo.add %6, %87 : tensor<1x128xf32>
    %89 = stablehlo.power %88, %1 : tensor<1x128xf32>
    %90 = stablehlo.reduce(%89 init: %4) across dimensions = [1] : (tensor<1x128xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg43: tensor<f32>, %arg44: tensor<f32>)  {
      %329 = stablehlo.add %arg43, %arg44 : tensor<f32>
      stablehlo.return %329 : tensor<f32>
    }
    %91 = stablehlo.multiply %90, %0 : tensor<1xf32>
    %92 = stablehlo.reshape %91 : (tensor<1xf32>) -> tensor<1x1xf32>
    %93 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %94 = stablehlo.add %92, %93 : tensor<1x1xf32>
    %95 = stablehlo.rsqrt %94 : tensor<1x1xf32>
    %96 = stablehlo.reshape %95 : (tensor<1x1xf32>) -> tensor<1xf32>
    %97 = stablehlo.broadcast_in_dim %96, dims = [0] : (tensor<1xf32>) -> tensor<1x128xf32>
    %98 = stablehlo.multiply %88, %97 : tensor<1x128xf32>
    %99 = stablehlo.reshape %arg18 : (tensor<128xf32>) -> tensor<1x128xf32>
    %100 = stablehlo.multiply %98, %99 : tensor<1x128xf32>
    %101 = stablehlo.transpose %arg32, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,352]{0,1}"} : (tensor<352x128xf32>) -> tensor<128x352xf32>
    %102 = stablehlo.dot %100, %101, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x352xf32>) -> tensor<1x352xf32>
    %103 = stablehlo.logistic %102 : tensor<1x352xf32>
    %104 = stablehlo.multiply %102, %103 : tensor<1x352xf32>
    %105 = stablehlo.transpose %arg17, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,352]{0,1}"} : (tensor<352x128xf32>) -> tensor<128x352xf32>
    %106 = stablehlo.dot %100, %105, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x352xf32>) -> tensor<1x352xf32>
    %107 = stablehlo.multiply %104, %106 : tensor<1x352xf32>
    %108 = stablehlo.transpose %arg16, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[352,128]{0,1}"} : (tensor<128x352xf32>) -> tensor<352x128xf32>
    %109 = stablehlo.dot %107, %108, precision = [DEFAULT, DEFAULT] : (tensor<1x352xf32>, tensor<352x128xf32>) -> tensor<1x128xf32>
    %110 = stablehlo.add %88, %109 : tensor<1x128xf32>
    %111 = stablehlo.power %110, %1 : tensor<1x128xf32>
    %112 = stablehlo.reduce(%111 init: %4) across dimensions = [1] : (tensor<1x128xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg43: tensor<f32>, %arg44: tensor<f32>)  {
      %329 = stablehlo.add %arg43, %arg44 : tensor<f32>
      stablehlo.return %329 : tensor<f32>
    }
    %113 = stablehlo.multiply %112, %0 : tensor<1xf32>
    %114 = stablehlo.reshape %113 : (tensor<1xf32>) -> tensor<1x1xf32>
    %115 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %116 = stablehlo.add %114, %115 : tensor<1x1xf32>
    %117 = stablehlo.rsqrt %116 : tensor<1x1xf32>
    %118 = stablehlo.reshape %117 : (tensor<1x1xf32>) -> tensor<1xf32>
    %119 = stablehlo.broadcast_in_dim %118, dims = [0] : (tensor<1xf32>) -> tensor<1x128xf32>
    %120 = stablehlo.multiply %110, %119 : tensor<1x128xf32>
    %121 = stablehlo.reshape %arg15 : (tensor<128xf32>) -> tensor<1x128xf32>
    %122 = stablehlo.multiply %120, %121 : tensor<1x128xf32>
    %123 = stablehlo.transpose %arg36, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %124 = stablehlo.dot %122, %123, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %125 = stablehlo.reshape %124 : (tensor<1x128xf32>) -> tensor<1x2x32x2xf32>
    %126 = stablehlo.slice %125 [0:1, 0:2, 0:32, 0:1] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x1xf32>
    %127 = stablehlo.reshape %126 : (tensor<1x2x32x1xf32>) -> tensor<1x2x32xf32>
    %128 = stablehlo.slice %125 [0:1, 0:2, 0:32, 1:2] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x1xf32>
    %129 = stablehlo.reshape %128 : (tensor<1x2x32x1xf32>) -> tensor<1x2x32xf32>
    %130 = stablehlo.complex %127, %129 : tensor<1x2x32xcomplex<f32>>
    %131 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x32xcomplex<f32>>) -> tensor<1x2x32xcomplex<f32>>
    %132 = stablehlo.multiply %130, %131 : tensor<1x2x32xcomplex<f32>>
    %133 = stablehlo.real %132 : (tensor<1x2x32xcomplex<f32>>) -> tensor<1x2x32xf32>
    %134 = stablehlo.reshape %133 : (tensor<1x2x32xf32>) -> tensor<1x2x32x1xf32>
    %135 = stablehlo.imag %132 : (tensor<1x2x32xcomplex<f32>>) -> tensor<1x2x32xf32>
    %136 = stablehlo.reshape %135 : (tensor<1x2x32xf32>) -> tensor<1x2x32x1xf32>
    %137 = stablehlo.concatenate %134, %136, dim = 3 : (tensor<1x2x32x1xf32>, tensor<1x2x32x1xf32>) -> tensor<1x2x32x2xf32>
    %138 = stablehlo.reshape %137 : (tensor<1x2x32x2xf32>) -> tensor<2x1x64xf32>
    %139 = stablehlo.compare  LT, %arg24, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %140 = stablehlo.reshape %arg25 : (tensor<i64>) -> tensor<1xi64>
    %141 = stablehlo.add %arg24, %140 : tensor<1xi64>
    %142 = stablehlo.select %139, %141, %arg24 : tensor<1xi1>, tensor<1xi64>
    %143 = stablehlo.reshape %142 : (tensor<1xi64>) -> tensor<1x1xi64>
    %144 = stablehlo.transpose %arg34, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %145 = stablehlo.dot %122, %144, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %146 = stablehlo.reshape %145 : (tensor<1x128xf32>) -> tensor<1x2x32x2xf32>
    %147 = stablehlo.slice %146 [0:1, 0:2, 0:32, 0:1] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x1xf32>
    %148 = stablehlo.reshape %147 : (tensor<1x2x32x1xf32>) -> tensor<1x2x32xf32>
    %149 = stablehlo.slice %146 [0:1, 0:2, 0:32, 1:2] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x1xf32>
    %150 = stablehlo.reshape %149 : (tensor<1x2x32x1xf32>) -> tensor<1x2x32xf32>
    %151 = stablehlo.complex %148, %150 : tensor<1x2x32xcomplex<f32>>
    %152 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x32xcomplex<f32>>) -> tensor<1x2x32xcomplex<f32>>
    %153 = stablehlo.multiply %151, %152 : tensor<1x2x32xcomplex<f32>>
    %154 = stablehlo.real %153 : (tensor<1x2x32xcomplex<f32>>) -> tensor<1x2x32xf32>
    %155 = stablehlo.reshape %154 : (tensor<1x2x32xf32>) -> tensor<1x2x32x1xf32>
    %156 = stablehlo.imag %153 : (tensor<1x2x32xcomplex<f32>>) -> tensor<1x2x32xf32>
    %157 = stablehlo.reshape %156 : (tensor<1x2x32xf32>) -> tensor<1x2x32x1xf32>
    %158 = stablehlo.concatenate %155, %157, dim = 3 : (tensor<1x2x32x1xf32>, tensor<1x2x32x1xf32>) -> tensor<1x2x32x2xf32>
    %159 = stablehlo.reshape %158 : (tensor<1x2x32x2xf32>) -> tensor<1x2x64xf32>
    %160 = "stablehlo.scatter"(%arg35, %143, %159) ({
    ^bb0(%arg43: tensor<f32>, %arg44: tensor<f32>):
      stablehlo.return %arg44 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x2x64xf32>, tensor<1x1xi64>, tensor<1x2x64xf32>) -> tensor<2304x2x64xf32>
    %161 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %162 = "stablehlo.gather"(%160, %161) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 2, 64]> : tensor<3xi64>} : (tensor<2304x2x64xf32>, tensor<2048xui32>) -> tensor<2048x2x64xf32>
    %163 = stablehlo.transpose %162, dims = [1, 2, 0] : (tensor<2048x2x64xf32>) -> tensor<2x64x2048xf32>
    %164 = stablehlo.dot_general %138, %163, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x1x64xf32>, tensor<2x64x2048xf32>) -> tensor<2x1x2048xf32>
    %165 = stablehlo.broadcast_in_dim %arg27, dims = [] : (tensor<f32>) -> tensor<2x1x2048xf32>
    %166 = stablehlo.divide %164, %165 : tensor<2x1x2048xf32>
    %167 = stablehlo.reduce(%166 init: %3) across dimensions = [2] : (tensor<2x1x2048xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg43: tensor<f32>, %arg44: tensor<f32>)  {
      %329 = stablehlo.maximum %arg43, %arg44 : tensor<f32>
      stablehlo.return %329 : tensor<f32>
    }
    %168 = stablehlo.broadcast_in_dim %167, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x2048xf32>
    %169 = stablehlo.subtract %166, %168 : tensor<2x1x2048xf32>
    %170 = stablehlo.exponential %169 : tensor<2x1x2048xf32>
    %171 = stablehlo.reduce(%170 init: %4) across dimensions = [2] : (tensor<2x1x2048xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg43: tensor<f32>, %arg44: tensor<f32>)  {
      %329 = stablehlo.add %arg43, %arg44 : tensor<f32>
      stablehlo.return %329 : tensor<f32>
    }
    %172 = stablehlo.broadcast_in_dim %171, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x2048xf32>
    %173 = stablehlo.divide %170, %172 : tensor<2x1x2048xf32>
    %174 = stablehlo.compare  LT, %arg24, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %175 = stablehlo.reshape %arg25 : (tensor<i64>) -> tensor<1xi64>
    %176 = stablehlo.add %arg24, %175 : tensor<1xi64>
    %177 = stablehlo.select %174, %176, %arg24 : tensor<1xi1>, tensor<1xi64>
    %178 = stablehlo.reshape %177 : (tensor<1xi64>) -> tensor<1x1xi64>
    %179 = stablehlo.transpose %arg14, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %180 = stablehlo.dot %122, %179, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %181 = stablehlo.reshape %180 : (tensor<1x128xf32>) -> tensor<1x2x64xf32>
    %182 = "stablehlo.scatter"(%arg33, %178, %181) ({
    ^bb0(%arg43: tensor<f32>, %arg44: tensor<f32>):
      stablehlo.return %arg44 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x2x64xf32>, tensor<1x1xi64>, tensor<1x2x64xf32>) -> tensor<2304x2x64xf32>
    %183 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %184 = "stablehlo.gather"(%182, %183) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 2, 64]> : tensor<3xi64>} : (tensor<2304x2x64xf32>, tensor<2048xui32>) -> tensor<2048x2x64xf32>
    %185 = stablehlo.transpose %184, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[2,2048,64]{2,0,1}"} : (tensor<2048x2x64xf32>) -> tensor<2x2048x64xf32>
    %186 = stablehlo.dot_general %173, %185, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x1x2048xf32>, tensor<2x2048x64xf32>) -> tensor<2x1x64xf32>
    %187 = stablehlo.reshape %186 : (tensor<2x1x64xf32>) -> tensor<1x128xf32>
    %188 = stablehlo.transpose %arg13, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %189 = stablehlo.dot %187, %188, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %190 = stablehlo.add %110, %189 : tensor<1x128xf32>
    %191 = stablehlo.power %190, %1 : tensor<1x128xf32>
    %192 = stablehlo.reduce(%191 init: %4) across dimensions = [1] : (tensor<1x128xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg43: tensor<f32>, %arg44: tensor<f32>)  {
      %329 = stablehlo.add %arg43, %arg44 : tensor<f32>
      stablehlo.return %329 : tensor<f32>
    }
    %193 = stablehlo.multiply %192, %0 : tensor<1xf32>
    %194 = stablehlo.reshape %193 : (tensor<1xf32>) -> tensor<1x1xf32>
    %195 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %196 = stablehlo.add %194, %195 : tensor<1x1xf32>
    %197 = stablehlo.rsqrt %196 : tensor<1x1xf32>
    %198 = stablehlo.reshape %197 : (tensor<1x1xf32>) -> tensor<1xf32>
    %199 = stablehlo.broadcast_in_dim %198, dims = [0] : (tensor<1xf32>) -> tensor<1x128xf32>
    %200 = stablehlo.multiply %190, %199 : tensor<1x128xf32>
    %201 = stablehlo.reshape %arg12 : (tensor<128xf32>) -> tensor<1x128xf32>
    %202 = stablehlo.multiply %200, %201 : tensor<1x128xf32>
    %203 = stablehlo.transpose %arg37, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,352]{0,1}"} : (tensor<352x128xf32>) -> tensor<128x352xf32>
    %204 = stablehlo.dot %202, %203, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x352xf32>) -> tensor<1x352xf32>
    %205 = stablehlo.logistic %204 : tensor<1x352xf32>
    %206 = stablehlo.multiply %204, %205 : tensor<1x352xf32>
    %207 = stablehlo.transpose %arg11, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,352]{0,1}"} : (tensor<352x128xf32>) -> tensor<128x352xf32>
    %208 = stablehlo.dot %202, %207, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x352xf32>) -> tensor<1x352xf32>
    %209 = stablehlo.multiply %206, %208 : tensor<1x352xf32>
    %210 = stablehlo.transpose %arg10, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[352,128]{0,1}"} : (tensor<128x352xf32>) -> tensor<352x128xf32>
    %211 = stablehlo.dot %209, %210, precision = [DEFAULT, DEFAULT] : (tensor<1x352xf32>, tensor<352x128xf32>) -> tensor<1x128xf32>
    %212 = stablehlo.add %190, %211 : tensor<1x128xf32>
    %213 = stablehlo.power %212, %1 : tensor<1x128xf32>
    %214 = stablehlo.reduce(%213 init: %4) across dimensions = [1] : (tensor<1x128xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg43: tensor<f32>, %arg44: tensor<f32>)  {
      %329 = stablehlo.add %arg43, %arg44 : tensor<f32>
      stablehlo.return %329 : tensor<f32>
    }
    %215 = stablehlo.multiply %214, %0 : tensor<1xf32>
    %216 = stablehlo.reshape %215 : (tensor<1xf32>) -> tensor<1x1xf32>
    %217 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %218 = stablehlo.add %216, %217 : tensor<1x1xf32>
    %219 = stablehlo.rsqrt %218 : tensor<1x1xf32>
    %220 = stablehlo.reshape %219 : (tensor<1x1xf32>) -> tensor<1xf32>
    %221 = stablehlo.broadcast_in_dim %220, dims = [0] : (tensor<1xf32>) -> tensor<1x128xf32>
    %222 = stablehlo.multiply %212, %221 : tensor<1x128xf32>
    %223 = stablehlo.reshape %arg9 : (tensor<128xf32>) -> tensor<1x128xf32>
    %224 = stablehlo.multiply %222, %223 : tensor<1x128xf32>
    %225 = stablehlo.transpose %arg41, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %226 = stablehlo.dot %224, %225, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %227 = stablehlo.reshape %226 : (tensor<1x128xf32>) -> tensor<1x2x32x2xf32>
    %228 = stablehlo.slice %227 [0:1, 0:2, 0:32, 0:1] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x1xf32>
    %229 = stablehlo.reshape %228 : (tensor<1x2x32x1xf32>) -> tensor<1x2x32xf32>
    %230 = stablehlo.slice %227 [0:1, 0:2, 0:32, 1:2] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x1xf32>
    %231 = stablehlo.reshape %230 : (tensor<1x2x32x1xf32>) -> tensor<1x2x32xf32>
    %232 = stablehlo.complex %229, %231 : tensor<1x2x32xcomplex<f32>>
    %233 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x32xcomplex<f32>>) -> tensor<1x2x32xcomplex<f32>>
    %234 = stablehlo.multiply %232, %233 : tensor<1x2x32xcomplex<f32>>
    %235 = stablehlo.real %234 : (tensor<1x2x32xcomplex<f32>>) -> tensor<1x2x32xf32>
    %236 = stablehlo.reshape %235 : (tensor<1x2x32xf32>) -> tensor<1x2x32x1xf32>
    %237 = stablehlo.imag %234 : (tensor<1x2x32xcomplex<f32>>) -> tensor<1x2x32xf32>
    %238 = stablehlo.reshape %237 : (tensor<1x2x32xf32>) -> tensor<1x2x32x1xf32>
    %239 = stablehlo.concatenate %236, %238, dim = 3 : (tensor<1x2x32x1xf32>, tensor<1x2x32x1xf32>) -> tensor<1x2x32x2xf32>
    %240 = stablehlo.reshape %239 : (tensor<1x2x32x2xf32>) -> tensor<2x1x64xf32>
    %241 = stablehlo.compare  LT, %arg24, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %242 = stablehlo.reshape %arg25 : (tensor<i64>) -> tensor<1xi64>
    %243 = stablehlo.add %arg24, %242 : tensor<1xi64>
    %244 = stablehlo.select %241, %243, %arg24 : tensor<1xi1>, tensor<1xi64>
    %245 = stablehlo.reshape %244 : (tensor<1xi64>) -> tensor<1x1xi64>
    %246 = stablehlo.transpose %arg39, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %247 = stablehlo.dot %224, %246, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %248 = stablehlo.reshape %247 : (tensor<1x128xf32>) -> tensor<1x2x32x2xf32>
    %249 = stablehlo.slice %248 [0:1, 0:2, 0:32, 0:1] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x1xf32>
    %250 = stablehlo.reshape %249 : (tensor<1x2x32x1xf32>) -> tensor<1x2x32xf32>
    %251 = stablehlo.slice %248 [0:1, 0:2, 0:32, 1:2] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x1xf32>
    %252 = stablehlo.reshape %251 : (tensor<1x2x32x1xf32>) -> tensor<1x2x32xf32>
    %253 = stablehlo.complex %250, %252 : tensor<1x2x32xcomplex<f32>>
    %254 = stablehlo.broadcast_in_dim %28, dims = [0, 2] : (tensor<1x32xcomplex<f32>>) -> tensor<1x2x32xcomplex<f32>>
    %255 = stablehlo.multiply %253, %254 : tensor<1x2x32xcomplex<f32>>
    %256 = stablehlo.real %255 : (tensor<1x2x32xcomplex<f32>>) -> tensor<1x2x32xf32>
    %257 = stablehlo.reshape %256 : (tensor<1x2x32xf32>) -> tensor<1x2x32x1xf32>
    %258 = stablehlo.imag %255 : (tensor<1x2x32xcomplex<f32>>) -> tensor<1x2x32xf32>
    %259 = stablehlo.reshape %258 : (tensor<1x2x32xf32>) -> tensor<1x2x32x1xf32>
    %260 = stablehlo.concatenate %257, %259, dim = 3 : (tensor<1x2x32x1xf32>, tensor<1x2x32x1xf32>) -> tensor<1x2x32x2xf32>
    %261 = stablehlo.reshape %260 : (tensor<1x2x32x2xf32>) -> tensor<1x2x64xf32>
    %262 = "stablehlo.scatter"(%arg40, %245, %261) ({
    ^bb0(%arg43: tensor<f32>, %arg44: tensor<f32>):
      stablehlo.return %arg44 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x2x64xf32>, tensor<1x1xi64>, tensor<1x2x64xf32>) -> tensor<2304x2x64xf32>
    %263 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %264 = "stablehlo.gather"(%262, %263) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 2, 64]> : tensor<3xi64>} : (tensor<2304x2x64xf32>, tensor<2048xui32>) -> tensor<2048x2x64xf32>
    %265 = stablehlo.transpose %264, dims = [1, 2, 0] : (tensor<2048x2x64xf32>) -> tensor<2x64x2048xf32>
    %266 = stablehlo.dot_general %240, %265, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x1x64xf32>, tensor<2x64x2048xf32>) -> tensor<2x1x2048xf32>
    %267 = stablehlo.broadcast_in_dim %arg27, dims = [] : (tensor<f32>) -> tensor<2x1x2048xf32>
    %268 = stablehlo.divide %266, %267 : tensor<2x1x2048xf32>
    %269 = stablehlo.reduce(%268 init: %3) across dimensions = [2] : (tensor<2x1x2048xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg43: tensor<f32>, %arg44: tensor<f32>)  {
      %329 = stablehlo.maximum %arg43, %arg44 : tensor<f32>
      stablehlo.return %329 : tensor<f32>
    }
    %270 = stablehlo.broadcast_in_dim %269, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x2048xf32>
    %271 = stablehlo.subtract %268, %270 : tensor<2x1x2048xf32>
    %272 = stablehlo.exponential %271 : tensor<2x1x2048xf32>
    %273 = stablehlo.reduce(%272 init: %4) across dimensions = [2] : (tensor<2x1x2048xf32>, tensor<f32>) -> tensor<2x1xf32>
     reducer(%arg43: tensor<f32>, %arg44: tensor<f32>)  {
      %329 = stablehlo.add %arg43, %arg44 : tensor<f32>
      stablehlo.return %329 : tensor<f32>
    }
    %274 = stablehlo.broadcast_in_dim %273, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1x2048xf32>
    %275 = stablehlo.divide %272, %274 : tensor<2x1x2048xf32>
    %276 = stablehlo.compare  LT, %arg24, %2 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %277 = stablehlo.reshape %arg25 : (tensor<i64>) -> tensor<1xi64>
    %278 = stablehlo.add %arg24, %277 : tensor<1xi64>
    %279 = stablehlo.select %276, %278, %arg24 : tensor<1xi1>, tensor<1xi64>
    %280 = stablehlo.reshape %279 : (tensor<1xi64>) -> tensor<1x1xi64>
    %281 = stablehlo.transpose %arg8, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %282 = stablehlo.dot %224, %281, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %283 = stablehlo.reshape %282 : (tensor<1x128xf32>) -> tensor<1x2x64xf32>
    %284 = "stablehlo.scatter"(%arg38, %280, %283) ({
    ^bb0(%arg43: tensor<f32>, %arg44: tensor<f32>):
      stablehlo.return %arg44 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2304x2x64xf32>, tensor<1x1xi64>, tensor<1x2x64xf32>) -> tensor<2304x2x64xf32>
    %285 = stablehlo.convert %arg7 : (tensor<2048xi64>) -> tensor<2048xui32>
    %286 = "stablehlo.gather"(%284, %285) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 2, 64]> : tensor<3xi64>} : (tensor<2304x2x64xf32>, tensor<2048xui32>) -> tensor<2048x2x64xf32>
    %287 = stablehlo.transpose %286, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[2,2048,64]{2,0,1}"} : (tensor<2048x2x64xf32>) -> tensor<2x2048x64xf32>
    %288 = stablehlo.dot_general %275, %287, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x1x2048xf32>, tensor<2x2048x64xf32>) -> tensor<2x1x64xf32>
    %289 = stablehlo.reshape %288 : (tensor<2x1x64xf32>) -> tensor<1x128xf32>
    %290 = stablehlo.transpose %arg6, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,128]{0,1}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %291 = stablehlo.dot %289, %290, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x128xf32>) -> tensor<1x128xf32>
    %292 = stablehlo.add %212, %291 : tensor<1x128xf32>
    %293 = stablehlo.power %292, %1 : tensor<1x128xf32>
    %294 = stablehlo.reduce(%293 init: %4) across dimensions = [1] : (tensor<1x128xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg43: tensor<f32>, %arg44: tensor<f32>)  {
      %329 = stablehlo.add %arg43, %arg44 : tensor<f32>
      stablehlo.return %329 : tensor<f32>
    }
    %295 = stablehlo.multiply %294, %0 : tensor<1xf32>
    %296 = stablehlo.reshape %295 : (tensor<1xf32>) -> tensor<1x1xf32>
    %297 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %298 = stablehlo.add %296, %297 : tensor<1x1xf32>
    %299 = stablehlo.rsqrt %298 : tensor<1x1xf32>
    %300 = stablehlo.reshape %299 : (tensor<1x1xf32>) -> tensor<1xf32>
    %301 = stablehlo.broadcast_in_dim %300, dims = [0] : (tensor<1xf32>) -> tensor<1x128xf32>
    %302 = stablehlo.multiply %292, %301 : tensor<1x128xf32>
    %303 = stablehlo.reshape %arg5 : (tensor<128xf32>) -> tensor<1x128xf32>
    %304 = stablehlo.multiply %302, %303 : tensor<1x128xf32>
    %305 = stablehlo.transpose %arg42, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,352]{0,1}"} : (tensor<352x128xf32>) -> tensor<128x352xf32>
    %306 = stablehlo.dot %304, %305, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x352xf32>) -> tensor<1x352xf32>
    %307 = stablehlo.logistic %306 : tensor<1x352xf32>
    %308 = stablehlo.multiply %306, %307 : tensor<1x352xf32>
    %309 = stablehlo.transpose %arg4, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,352]{0,1}"} : (tensor<352x128xf32>) -> tensor<128x352xf32>
    %310 = stablehlo.dot %304, %309, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x352xf32>) -> tensor<1x352xf32>
    %311 = stablehlo.multiply %308, %310 : tensor<1x352xf32>
    %312 = stablehlo.transpose %arg3, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[352,128]{0,1}"} : (tensor<128x352xf32>) -> tensor<352x128xf32>
    %313 = stablehlo.dot %311, %312, precision = [DEFAULT, DEFAULT] : (tensor<1x352xf32>, tensor<352x128xf32>) -> tensor<1x128xf32>
    %314 = stablehlo.add %292, %313 : tensor<1x128xf32>
    %315 = stablehlo.power %314, %1 : tensor<1x128xf32>
    %316 = stablehlo.reduce(%315 init: %4) across dimensions = [1] : (tensor<1x128xf32>, tensor<f32>) -> tensor<1xf32>
     reducer(%arg43: tensor<f32>, %arg44: tensor<f32>)  {
      %329 = stablehlo.add %arg43, %arg44 : tensor<f32>
      stablehlo.return %329 : tensor<f32>
    }
    %317 = stablehlo.multiply %316, %0 : tensor<1xf32>
    %318 = stablehlo.reshape %317 : (tensor<1xf32>) -> tensor<1x1xf32>
    %319 = stablehlo.reshape %arg2 : (tensor<f32>) -> tensor<1x1xf32>
    %320 = stablehlo.add %318, %319 : tensor<1x1xf32>
    %321 = stablehlo.rsqrt %320 : tensor<1x1xf32>
    %322 = stablehlo.reshape %321 : (tensor<1x1xf32>) -> tensor<1xf32>
    %323 = stablehlo.broadcast_in_dim %322, dims = [0] : (tensor<1xf32>) -> tensor<1x128xf32>
    %324 = stablehlo.multiply %314, %323 : tensor<1x128xf32>
    %325 = stablehlo.reshape %arg1 : (tensor<128xf32>) -> tensor<1x128xf32>
    %326 = stablehlo.multiply %324, %325 : tensor<1x128xf32>
    %327 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[128,32000]{0,1}"} : (tensor<32000x128xf32>) -> tensor<128x32000xf32>
    %328 = stablehlo.dot %326, %327, precision = [DEFAULT, DEFAULT] : (tensor<1x128xf32>, tensor<128x32000xf32>) -> tensor<1x32000xf32>
    return %328, %58, %80, %160, %182, %262, %284 : tensor<1x32000xf32>, tensor<2304x2x64xf32>, tensor<2304x2x64xf32>, tensor<2304x2x64xf32>, tensor<2304x2x64xf32>, tensor<2304x2x64xf32>, tensor<2304x2x64xf32>
  }
}
