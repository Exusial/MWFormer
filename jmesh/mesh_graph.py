import jittor as jt

class MeshGraph:
    """
    MeshGraph is used to represent Graph-like mesh. If input meshes are not standard please use MeshGraph instead of MeshTensor. Their behavior is similar.
    """
    def __init__(self, feats: jt.Var, Fs: jt.Var=None, Vertices: jt.Var=None, next_size=None, cache=None):
        self.feats = feats
        self.N, self.C, self.F = feats.shape

        if Fs is not None:
            self.Fs = Fs 
            assert self.F == self.Fs.max().data[0]
        else:
            self.Fs = jt.ones(self.N, dtype="int32") * self.F
        self.graph_like = graph_like
        self.Vertices = Vertices
        if next_size is None:
            self.next_size = self.F
        else:
            self.next_size  = next_size
        self._cache = cache if cache is not None else {}

    def updated(self, new_feats):
        assert new_feats.shape[0] == self.N
        assert new_feats.shape[2] == self.F
        return MeshGraph(new_feats, self.Fs, self.Vertices, self._cache)

    @property
    def shape(self):
        return self.feats.shape

    @property
    def Vs(self) -> int:
        """ number of vertices in the meshgraph """
        return self.F
    
    @property
    def FAF(self) -> jt.Var:
        if not 'FAF' in self._cache:
            self._cache['FAF'] = None 
        return self._cache['FAF']

    # @property
    # def FAFP(self) -> jt.Var:
    #     """ The previous face of current face's adjacent faces """
    #     if not 'FAFP' in self._cache:
    #         self._cache['FAFP'], self._cache['FAFN'] = self.compute_face_adjacency_reordered()
    #     return self._cache['FAFP']

    # @property
    # def FAFN(self) -> jt.Var:
    #     """ The next face of current face's adjacent faces """
    #     if not 'FAFN' in self._cache:
    #         self._cache['FAFP'], self._cache['FAFN'] = self.compute_face_adjacency_reordered()
    #     return self._cache['FAFN']

    def __add__(self, other: jt.Var) -> jt.Var:
        new_feats = self.feats + other.feats
        return self.updated(new_feats)

    def __radd__(self, other: jt.Var) -> jt.Var:
        return self.__add__(other)

    def __sub__(self, other: jt.Var) -> jt.Var:
        new_feats = self.feats - other.feats
        return self.updated(new_feats)

    def __rsub__(self, other: jt.Var) -> jt.Var:
        new_feats = other.feats - self.feats
        return self.updated(new_feats)

    def __repr__(self):
        return f'MeshTensor: N={self.N}, C={self.C}, F={self.F}'

    def loop_unpool(self, mode, ref_faces=None, ref_cache=None):
        """
        Graph Pooling needs 
        """
        unpooled_Fs = self.Fs * 4

        if ref_faces is not None:
            unpooled_faces = ref_faces
            unpooled_cache = ref_cache
        else:
            unpooled_faces = self.loop_subdivision()
            unpooled_cache = None

        if mode == 'nearest':
            unpooled_feats = jt.concat([self.feats] * 4, dim=2)
        elif mode == 'bilinear':
            neighbor_feats = self.feats.reindex(
                shape=[self.N, self.C, self.F, 3],
                indexes=[
                    'i0', 'i1', '@e0(i0, i2, i3)'
                ],
                extras=[self.FAF]
            )
            unpooled_feats = jt.concat([
                (self.feats * 2 + neighbor_feats[..., 1] + neighbor_feats[..., 2]) / 4,
                (self.feats * 2 + neighbor_feats[..., 2] + neighbor_feats[..., 0]) / 4,
                (self.feats * 2 + neighbor_feats[..., 0] + neighbor_feats[..., 1]) / 4,
                self.feats
            ], dim=2)
        else:
            raise Exception(f'Unsupported unpool mode: {mode}')

        return MeshTensor(unpooled_faces, unpooled_feats, unpooled_Fs, unpooled_cache)

    def compute_adjacency(self, sFAF, res_faces, mode='sort') -> jt.Var:
        reduce_temp = -jt.ones_like(sFAF)
        reduce_adjaceny = reduce_temp.reindex_reduce(
            op='add',
            shape=[self.N, self.F, self.next_size],
                indexes=[
                'i0',
                '@e0(i1)',
                '@e0(i2)'
            ],
            extras=[self.res_faces, self.sFAF],
            overflow_conditions=['@e1(i0, i1, i2) == -1']
        )
        if mode == "sort":
            FAF, FAF_value = jt.argsort(reduce_adjaceny)
            FAF[FAF_value == 1] = -1
        else:
            pass # TODO: only need index.
        return FAF[:,:,:self.next_size]
    
    def compute_k_nearest_points(self):
        assert "FAF" in self.cache and self.cache["FAF"] is not None
        pass
    
    def dilated_face_adjacencies(self, dilation: int):
        if dilation <= 1:
            raise Exception('dilation must be greater than zero')

        DFA = jt.code(
            shape=[self.N, self.F, 3],
            dtype=jt.int32,
            inputs=[self.FAF, jt.zeros((dilation, 0), dtype=jt.int32)],
            cpu_src="""
                @alias(FAF, in0)
                int dilation = in1_shape0;

                for (int bs = 0; bs < out_shape0; ++bs)
                    for (int f = 0; f < out_shape1; ++f)
                        for (int k = 0; k < out_shape2; ++k) {
                            int a = f;
                            int b = @FAF(bs, f, k);
                            for (int d = 1; d < dilation; ++d) {
                                int i = @FAF(bs, b, 0) == a ? 0 : (@FAF(bs, b, 1) == a ? 1 : 2);
                                a = b;
                                if ((d & 1) == 0) {       // go to next
                                    b = @FAF(bs, b, i < 2 ? i + 1 : 0);
                                } else {                // go to previous
                                    b = @FAF(bs, b, i > 0 ? i - 1 : 2);
                                }
                            }
                            @out(bs, f, k) = b;
                        }
            """,
            cuda_src="""
                __global__ void dilated_face_adjacencies_kernel(@ARGS_DEF) {
                    @PRECALC
                    @alias(FAF, in0)
                    int dilation = in1_shape0;
                    int N = in0_shape0;
                    int F = in0_shape1;

                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    int bs = idx / (F * 3);
                    int f = idx / 3 % F;
                    int k = idx % 3;

                    if (bs >= N)
                        return;

                    int a = f;
                    int b = @FAF(bs, f, k);
                    for (int d = 1; d < dilation; ++d) {
                        int i = @FAF(bs, b, 0) == a ? 0 : (@FAF(bs, b, 1) == a ? 1 : 2);
                        a = b;
                        if ((d & 1) == 0) {     // go to next
                            b = @FAF(bs, b, i < 2 ? i + 1 : 0);
                        } else {                // go to previous
                            b = @FAF(bs, b, i > 0 ? i - 1 : 2);
                        }
                    }
                    @out(bs, f, k) = b;
                }

                dilated_face_adjacencies_kernel<<<(in0_shape0*in0_shape1*3-1)/512+1, 512>>>(@ARGS);
            """
        )

        return DFA

    def convolution_kernel_pattern(self, kernel_size=3, dilation=1):
        if kernel_size == 1:
            raise Exception(f'kernel size 1 does not have convolution pattern')

        if kernel_size == 3:
            if dilation == 1:
                return self.FAF
            else:
                return self.dilated_face_adjacencies(dilation)
        elif kernel_size == 5:
            if dilation == 1:
                return jt.stack([
                    self.FAFN[:, :, 0],
                    self.FAF[:, :, 0],
                    self.FAFP[:, :, 0],
                    self.FAFN[:, :, 1],
                    self.FAF[:, :, 1],
                    self.FAFP[:, :, 1],
                    self.FAFN[:, :, 2],
                    self.FAF[:, :, 2],
                    self.FAFP[:, :, 2],
                ], dim=-1)
            else:
                raise Exception('Not support dilation with kernel size larger than 3 yet')
        else:
            DFA = jt.code(
                shape=[self.N, self.F, 3],
                dtype=jt.int32,
                inputs=[self.FAF, jt.zeros(kernel_size, 0), jt.zeros((dilation, 0), dtype=jt.int32)],
                cpu_src="""
                    @alias(FAF, in0)
                    int kernel_size = in1_shape0;
                    int dilation = in2_shape0;

                    for (int bs = 0; bs < out_shape0; ++bs)
                        for (int f = 0; f < out_shape1; ++f)
                            for (int k = 0; k < out_shape2; ++k) {
                                int a = f;
                                int b = @FAF(bs, f, k);
                                for (int d = 1; d < 0; ++d) {
                                    int i = @FAF(bs, b, 0) == a ? 0 : (@FAF(bs, b, 1) == a ? 1 : 2);
                                    a = b;
                                    if ((d & 1) == 0) {       // go to next
                                        b = @FAF(bs, b, i < 2 ? i + 1 : 0);
                                    } else {                // go to previous
                                        b = @FAF(bs, b, i > 0 ? i - 1 : 2);
                                    }
                                }
                                @out(bs, f, k) = b;
                            }
                """,
                cuda_src="""
                    __global__ void dilated_face_adjacencies_kernel(@ARGS_DEF) {
                        @PRECALC
                        @alias(FAF, in0)
                        int dilation = in1_shape0;
                        int N = in0_shape0;
                        int F = in0_shape1;

                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        int bs = idx / (F * 3);
                        int f = idx / 3 % F;
                        int k = idx % 3;

                        if (bs >= N)
                            return;

                        int a = f;
                        int b = @FAF(bs, f, k);
                        for (int d = 1; d < dilation; ++d) {
                            int i = @FAF(bs, b, 0) == a ? 0 : (@FAF(bs, b, 1) == a ? 1 : 2);
                            a = b;
                            if ((d & 1) == 0) {     // go to next
                                b = @FAF(bs, b, i < 2 ? i + 1 : 0);
                            } else {                // go to previous
                                b = @FAF(bs, b, i > 0 ? i - 1 : 2);
                            }
                        }
                        @out(bs, f, k) = b;
                    }

                    dilated_face_adjacencies_kernel<<<(in0_shape0*in0_shape1*3-1)/512+1, 512>>>(@ARGS);
                """
            )

            return DFA

            raise Exception(f'Unspported kernel size {kernel_size}')