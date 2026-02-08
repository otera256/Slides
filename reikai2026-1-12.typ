#import "@preview/touying:0.6.1": *
#import themes.metropolis: *
#import "@preview/numbly:0.1.0": numbly
#import "@preview/algorithmic:1.0.7"
#import algorithmic: algorithm-figure, style-algorithm
#show: style-algorithm
#import "@preview/theorion:0.4.1": *

// --- 図解用ライブラリのインポート ---
#import "@preview/fletcher:0.5.8": diagram, node, edge, shapes
#import "@preview/lilaq:0.4.0" as lq

// --- 設定 ---
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.institution,
  config-info(
    title: [BevyとWGSLでマンデルブロ集合を描画する],
    author: [49th furakuta],
    date: datetime.today(),
    institution: [KMC],
  )
)

#set text(lang: "ja", font: "UD Digi Kyokasho NP")
#set heading(numbering: numbly("{1}.", default: "1.1"))

// 数式ブロックのスタイル
#show math.equation.where(block: true): it => block(inset: 10pt, radius: 5pt, fill: rgb("#f0f0f0"), width: 100%, it)

#let image-slide(body, background: none) = touying-slide-wrapper(self => {
  self = utils.merge-dicts(
    self,
    config-page(
      background: place(right, background),
      margin: 0em,
    ),
  )
  set image(width: auto, height: auto)
  touying-slide(self: self, align(bottom, body))
})

#title-slide()

#focus-slide[とりあえず完成品を見てみよう]

= マンデルブロ集合とは

== 定義

次の漸化式
$
  cases(
    z_(n + 1) = z_n^2 + c,
    z_0 = 0
  )
$
で定義される複素数列${z_n}_(n = 0)^infinity$について，初項$c$に対して${z_n}$が発散しないとき，$c$はマンデルブロ集合に属すると定義する．

複素数$c = a + b i$を平面上の点$(a, b)$に対応させることで，マンデルブロ集合は平面上の点の集合として描画できる．

#text(size: 10pt, fill: luma(50%))[
  参考: ウィキペディア マンデルブロ集合 https://ja.wikipedia.org/wiki/%E3%83%9E%E3%83%B3%E3%83%87%E3%83%AB%E3%83%96%E3%83%AD%E9%9B%86%E5%90%88
]

== 単純なアルゴリズム

#show: style-algorithm
#algorithm-figure("最も単純なマンデルブロ集合描画アルゴリズム", {
  import algorithmic: *
  Assign[$z$][$0$]
  LineComment(Assign[$c$][$a + b i$], [平面上の位置に対応])  
  LineComment(Assign[$e$][$2$], [発散判定の閾値])
  LineComment(Assign[$N$][$100$], [最大繰り返し回数])
  For($n = 0 ;n < N; n++$, {
    Assign[$z$][$z^2 + c$]
    If( $|z| > e$ , {
      Return[white]
    })
  })
  Return[black]
})

#remark[
  初めの定義では${z_n}$の項が発散するとしていたが、$abs(z_n) > 2$となるような項が一つでもあれば$abs(z_n)$は必ず無限大に発散するということが知られているため，このように判定している．
]

#image-slide(
  image("images/mandelbrot_white_black.png", height: 100%)
)

= 利用した技術の詳細

== GPUによる高速化

マンデルブロ集合を描画するためには，各点に対して上記のアルゴリズムを実行する必要がある\
$->$ *計算量が非常に大きい*\
そのためGPUを用いた並列化による高速化を行う

=== *GPUとは？*
GPU(Graphics Processing Unit)は，CGやゲームなどのグラフィックス処理を高速に行うための専用プロセッサ


グラフィック処理においても画面上の各ピクセルに対して同じ処理を繰り返し行う必要があるため，このような並列処理に特化した設計となっている。ちなみに近年はその並列計算能力を生かしてAI用途などにも利用されている。このような用途をGPGPU(General-Purpose computing on Graphics Processing Units)と呼ぶ。

== そもそもなんでゲームエンジンを使っているのか
私はBevyというRust製のゲームエンジンを用いてWGSLでシェーダーを書きGPU上でマンデルブロ集合の描画を行った

=== Bevyとは？
BevyはRustで書かれたオープンソースのシンプルなデータ志向のゲームエンジンであり、以下のような特徴を持つ

- Rustで書かれていることによるメモリ安全性と高速性、強力な型システム
- データ(Component)と処理を分離して管理するEntity Component System(ECS)
- 2Dと3Dの両方に対応
- WASMも含めた広いクロスプラットフォーム性
- コミュニティ主導のエコシステム

#pagebreak()

今回の実装ではマウス操作によりインタラクティブにズームや移動ができるようにしたが，これらの機能を一から実装するのは大変

Bevyはゲームエンジンなのでマウス等の入力機能が初めから備わっており、さらにシェーダーを呼び出すための機能も提供されているため直接グラフィックAPIを操作するよりも簡単に実装できる

#strike[正直に言うとBevyを布教するのが目的]

#pagebreak()

== WGSLとは?

WGSL(WebGPU Shader Language)はWeb GPUを扱うためのシェーダー言語\
WebGPUはWebブラウザ上でGPUを利用するための新しいAPIであり，ブラウザでネイティブに近いパフォーマンスでグラフィックスや計算処理を実行できるようにすることを目的としている

- *ここでなんでWebの話が出るのか*
  - WebGPUはWebブラウザ上でGPUを利用するための新しいAPIであり，WGSLはそのシェーダー言語として設計された
  - WebGPU APIの主要な実装の一つはwgpuというRust製のライブラリであり，Bevyもこのライブラリを利用している

#strike()[正直GLSLみたいなメジャーな言語じゃないせいで情報が少なくて苦労した]

== 実装のポイント

- Bevyでは`Material`トレイトまたは`Material2D`を実装することで独自のマテリアル(シェーダー)を定義できる
- マテリアルに必要なパラメータ(ズーム率や中心座標など)をフィールドとして持たせ，`fragment_shader`メソッドでWGSLシェーダーのパスを指定する
- シェーダー内では各ピクセルの位置情報をもとにマンデルブロ集合のアルゴリズムを実行し，色を決定する
- 画面全体を覆う長方形を用意して、そのマテリアルを適用することで描画を行う

#pagebreak()
```rust
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct MandelbrotMaterial {
    #[uniform(0)]
    offset: Vec2, // 複素平面上の中心座標 (WGSL側では vec2<f32> になる)
    #[uniform(0)]
    range: f32,   // 複素平面上の表示範囲の高さ (WGSL側では f32 になる)
    #[uniform(0)]
    ratio: f32,   // ウィンドウのアスペクト比 (WGSL側では f32 になる)
}

// Material2dトレイトを実装して、どのシェーダーファイルを使うか指定します。
impl Material2d for MandelbrotMaterial {
    fn fragment_shader() -> ShaderRef {
        // assets/shaders/mandelbrot.wgsl を読み込む設定
        "shaders/mandelbrot.wgsl".into()
    }
}
```

```rust
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<MandelbrotMaterial>>, // 自作マテリアル用のアセットリソース
) {
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::default())),
        ScreenQuad,
        MeshMaterial2d(materials.add(MandelbrotMaterial {
            offset: Vec2::new(-0.5, 0.0), // 初期位置を少し左にずらす
            range: 4.0,
            ratio: 1.0, // 初期値（ウィンドウサイズに合わせて更新されます）
        })),
        Transform::from_xyz(0.0, 0.0, -1.0),
    ));
}
```
#pagebreak()
== GPUに値を渡す(Uniforms)
Bevyではマテリアルのバインドに基本的に`@group(0)`が使用され、そのなかでどこにパラメータを配置するかを`#[uniform(n)]`属性で指定する

WGSL側では`@group(0) @binding(n)`で対応するパラメータを受け取ることができる

```wgsl
struct MaterialData {
    offset: vec2<f32>,
    range: f32,
    ratio: f32,
};

@group(2) @binding(0) var<uniform> material: MaterialData;
```
#pagebreak()
```wgsl
const MAX_ITER: u32 = 100u;
const ESCAPE_RADIUS: f32 = 10000.0;

// マンデルブロ集合の点が発散するかどうかを判定します
// 発散までにかかる反復回数を返します
fn mandelbrot(c: vec2<f32>, z0: vec2<f32>) -> u32 {
    var z = z0;
    var i = 0u;
    for (; i < MAX_ITER; i ++) {
        if length(z) > ESCAPE_RADIUS {
            break;
        }
        // z <- z**2 + c
        z = vec2(z.x*z.x - z.y*z.y + c.x, 2*z.x*z.y + c.y);
    }
    return i;
}
```

```wgsl
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // 画素の位置を取得: (0, 0) が左上、(1, 1) が右下のUV座標系
    let uv: vec2<f32> = in.uv;
    // UV座標を複素平面上の座標に変換
    // material.offset が中心座標、material.range が表示範囲の高さ、material.ratio がアスペクト比
    let aspect_ratio = material.ratio;
    let c = vec2(
        material.offset.x + (uv.x - 0.5) * material.range * aspect_ratio,
        material.offset.y - (uv.y - 0.5) * material.range
    );
    // マンデルブロ集合の反復回数を計算
    let iter = mandelbrot(c, vec2(0.0, 0.0));
    if (iter == MAX_ITER) { return vec4(0.0, 0.0, 0.0, 1.0); } // 黒
    else { return vec4(1.0, 1.0, 1.0, 1.0); } // 白
}
```

= もっと見た目をよくしたい！

== カラーマッピング

さっきの実装では発散までの反復回数に関わらず白黒で描画していたが，反復回数に応じて色を変えることでより美しい画像にできる
```wgsl
let iter = mandelbrot(c, vec2(0.0, 0.0));

// 色を決定（少し見やすく調整）
if (iter == MAX_ITER) {
  return vec4(0.0, 0.0, 0.0, 1.0); // 収束したら黒
}
let color_value = f32(iter) / f32(MAX_ITER);
return vec4(color_value, 1.0);
```

#image-slide[
  #image("images/mandelbrot_before_smooth.png", height: 100%)
]

== スムーズカラーリング

前のスライドの画像では色の境界に段差があったが，発散までの反復回数に小数点以下の補正を加えることで滑らかにできる

具体的には$abs(z_n)$が十分に大きいときは$z_n = z_(n - 1)^2 + c$の$c$の項の影響は無視できるとして、$abs(z_n) = "ESCAPE_RADIUS"$をみたす$n$の値を小数点部分まで計算する. 求める値を$nu$とおくと
$
  z_n &= z_(n - 1)^2 \
  log_2|z_n| &= log_2|z_(n - 1)^2| = 2 log_2|z_(n - 1)| \
  log_2(log_2|z_n|) &= log_2(2 log_2|z_(n - 1)|) = 1 + log_2(log_2|z_(n - 1)|) \
  n - (n - 1) &= log_2(log_2|z_n|) - log_2(log_2|z_(n - 1)|) \
  n - nu &= log_2(log_2|z_n|) - log_2(log_2("ESCAPE_RADIUS")) \
  nu &= n - log_2(log_2|z_n|) + log_2(log_2("ESCAPE_RADIUS"))
$

ただし実装上は$log_2(log_2("ESCAPE_RADIUS"))$の項は定数として、適当な値（以下のコードでは1）を用いている
```wgsl
fn mandelbrot(c: vec2<f32>, z0: vec2<f32>) -> f32 {
    var z = z0;
    for (var i = 0u; i < MAX_ITER; i ++) {
        z = vec2(z.x*z.x - z.y*z.y, 2*z.x*z.y) + c; // z <- z**2 + c
        let z_len2 = dot(z, z);  // 計算コストを抑えるために長さの二乗を使う
        if z_len2 > ESCAPE_RADIUS * ESCAPE_RADIUS {
            let log_zn = log2(z_len2) / 2.0;
            return f32(i) + 1.0 - log2(log_zn);
        }
    }
    return f32(MAX_ITER);
}
```

#remark[
  近似する際に$abs(z_n)$が十分大きいという条件を満たすようにするために、ESCAPE_RADIUSやMAX_ITERを大きめに設定している
]

#image-slide[
  #image("images/mandelbrot_smooth.png", height: 100%)
]

== カラーマップの適用

発散までの反復回数に応じて色を変える際に，単純にグレースケールにするのではなくカラーマップを適用することでより美しい画像にできる
```wgsl
let iter = mandelbrot(c, vec2(0.0, 0.0));
// 色を決定（カラーマップを適用）
if (iter == MAX_ITER) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0); // 収束したら黒
}
let color_value = sin(
  (vec3f(f32(iter)/f32(MAX_ITER)) * vec3f(0.5, 2.5, 3.5)// RGBごとに周波数を変える
  +vec3f(0.5))*PI);
return vec4(color_value, 1.0);
```

#image-slide[
  #image("images/mandelbrot_color.png", width: 100%)
]

== アンチエイリアス

スライドでは見づらいかもしれないが、そのままの状態では境界部分のギザギザが目立つ\
これを改善するために、1ピクセル内で複数のサンプルを取得して平均化するアンチエイリアス処理を行う

```wgsl
fn get_color(c: vec2<f32>, z0: vec2<f32>) -> vec4<f32> {
  // ここでこれまでと同様にマンデルブロ集合の反復計算と色決定を行う
}
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // 画素の位置を取得して複素平面上の座標に変換
    let uv: vec2<f32> = in.uv;
    let aspect_ratio = material.aspect_ratio;
    let c0 = vec2(
        material.offset.x + (uv.x - 0.5) * material.range * aspect_ratio,
        material.offset.y - (uv.y - 0.5) * material.range
    );
    // アンチエイリアスのために4サンプルを取得して平均化
    let pixel_size = material.pixel_size; // 1ピクセルあたりの複素平面上の大きさ
    let samples = array<vec2<f32>, 4>(
        vec2(-0.25 * pixel_size, -0.25 * pixel_size),
        vec2( 0.25 * pixel_size, -0.25 * pixel_size),
        vec2(-0.25 * pixel_size,  0.25 * pixel_size),
        vec2( 0.25 * pixel_size,  0.25 * pixel_size),
    );
    var color = vec4(0.0);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let c = c0 + samples[i];
        color = color + get_color(c, vec2(0.0, 0.0));
    }
    color = color / 4.0;
    return color;
}
```

#image-slide[
  #image("images/mandelbrot_aa.png", width: 100%)
]

正直違いが判りずらいと思いますが、ディスプレイで見ると境界部分のギザギザがかなり軽減されていることがわかります。

またコードを見ての通り計算コストが4倍になるため、パフォーマンスとのトレードオフを考慮する必要があります。

= ズームしたい！！

== インタラクティブなズームと移動
正直ここまでは決まった位置のマンデルブロ集合を描画するだけだったので、プログラムを実行するたびに同じ画像が出力されるだけだった（もちろんパラメータを変えれば違う画像になるが）

そこでマウス操作でインタラクティブにズームや移動ができるようにし、マンデルブロ集合の探索を楽しめるようにした

結構楽しいのでぜひ試してみてください！

まずマテリアルのバインディングの実装を再掲する

#pagebreak()

```rust
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone, Default)]
struct MandelbrotMaterial {
    #[uniform(0)]
    offset: Vec2, // 複素平面上の中心座標 (WGSL側では vec2<f32> になる)
    #[uniform(0)]
    range: f32,  // 複素平面上の表示範囲の高さ (WGSL側では f32 になる)
    #[uniform(0)]
    aspect_ratio: f32,   // ウィンドウのアスペクト比 (WGSL側では f32 になる)
    #[uniform(0)]
    pixel_size: f32,   // 1ピクセルあたりの複素平面上の距離 (WGSL側では f32 になる)
}
impl Material2d for MandelbrotMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/mandelbrot.wgsl".into() // assets/shaders/mandelbrot.wgsl を読み込む設定
    }
}
```
```rust
impl MandelbrotMaterial {
    // マテリアルのパラメータを一括で更新するメソッド
    fn update_params(&mut self, params: &MandelbrotParams) {
        self.offset = params.offset;
        self.range = params.range;
        self.aspect_ratio = params.aspect_ratio();
        self.pixel_size = params.pixel_size();
    }
}
// いちいちmaterialにアクセスしないといけないのは面倒なので、CPU側の処理をまとめるためのResourceを用意する
#[derive(Resource, Debug, Clone)] // ResourceはBevyでのグローバル変数のようなもの
struct MandelbrotParams {
    offset: Vec2,
    range: f32,
    window_size: Vec2,
}
```
```rust
// マテリアルのパラメータを時間経過で更新するシステム
fn update_material(
    params: Res<MandelbrotParams>,
    // 現在シーンで使われている MandelbrotMaterial のハンドルを探すクエリ
    material_handle_query: Query<&MeshMaterial2d<MandelbrotMaterial>>,
    // マテリアルの実体データが格納されているアセットストレージへの可変アクセス
    mut material_assets: ResMut<Assets<MandelbrotMaterial>>,
) {
    // シーンにマテリアルがなければ何もしない
    let Ok(material_handle) = material_handle_query.single() else { return; };

    // ハンドルを使って、アセットストレージから実際のマテリアルデータを取得（可変）
    if let Some(material) = material_assets.get_mut(material_handle) {
        material.update_params(&params);
    }
}
```

```rust
fn zoom(
    mut msgr_scroll: MessageReader<MouseWheel>,
    window: Single<&Window>,
    mut mandelbrot_params: ResMut<MandelbrotParams>,
){
    use bevy::input::mouse::MouseScrollUnit;
    let Some(mouse_pos) = window.cursor_position() else { return; };
    let world_mouse_pos = Vec2::new(
        (mouse_pos.x / window.width() - 0.5) * mandelbrot_params.range * mandelbrot_params.aspect_ratio() + mandelbrot_params.offset.x,
        (0.5 - mouse_pos.y / window.height()) * mandelbrot_params.range + mandelbrot_params.offset.y,
    );
    for msg in msgr_scroll.read() {
        let scroll_amount = match msg.unit {
            MouseScrollUnit::Line => msg.y * 0.1,
            MouseScrollUnit::Pixel => msg.y * 0.001,
        };
        let zoom_factor = 1.0 - scroll_amount;
        mandelbrot_params.range *= zoom_factor;
        mandelbrot_params.offset = world_mouse_pos + (mandelbrot_params.offset - world_mouse_pos) * zoom_factor;
    }
}

fn drag(
    mut msgr_cursor: MessageReader<CursorMoved>,
    window: Single<&Window>,
    butttons: Res<ButtonInput<MouseButton>>,
    mut mandelbrot_params: ResMut<MandelbrotParams>,
){
    // マウス左ボタンが押されていない場合は何もしない
    if !butttons.pressed(MouseButton::Left) {
        return;
    }
    for msg in msgr_cursor.read() {
        let Some(delta) = msg.delta else { continue; };
        mandelbrot_params.offset.x -= delta.x / window.width() * mandelbrot_params.range * mandelbrot_params.aspect_ratio();
        mandelbrot_params.offset.y += delta.y / window.height() * mandelbrot_params.range;
    }
}
```

#pagebreak()

```rust
fn main() {
    App::new()
        .add_plugins((
            // Bevyのデフォルトプラグインを追加、画面を表示したりするのにこれが必要
            DefaultPlugins,
            // カスタムマテリアルを使うためのプラグインを追加
            // ResMut<Assets<MandelbrotMaterial>>とかが使えるのはこいつのおかげ
            Material2dPlugin::<MandelbrotMaterial>::default()
        ))
        // CPU側でマンデルブロ集合のパラメータを管理するためのResourceを初期化
        .init_resource::<MandelbrotParams>()
        .add_systems(Startup, (setup, resize_quad_to_window).chain())
        .add_systems(PreUpdate, (
            // ウィンドウサイズが変わったときに四角形のサイズを更新するシステム
            resize_quad_to_window.run_if(on_message::<WindowResized>),
        ))
        .add_systems(Update, (
            // 入力情報をもとにMandelbrotParamsを更新するシステム
            zoom,
            drag,
            // MandelbrotParamsの内容をマテリアルに反映するシステム
            update_material,
        ))
        .run();
}
```

ここまでの流れを要約すると
1. マテリアルのパラメータを保持する`MandelbrotParams`リソースを用意する
2. マウス入力を処理して`MandelbrotParams`を更新する
3. `MandelbrotParams`の内容をマテリアルに反映するシステムを追加する
4. マテリアル内のWGSLシェーダーでパラメータを受け取り、マンデルブロ集合の描画を行う

= ギャラリー

#image-slide[
  #image("images/gallery1.png", width: 100%)
]

#image-slide[
  #image("images/gallery2.png", width: 100%)
]

#image-slide[
  #image("images/gallery3.png", width: 100%)
]

#image-slide[
  #image("images/gallery4.png", width: 100%)
]

= もっとズームしたい！！！

== 単精度浮動小数点の限界

これまでのプログラムではすべて単精度浮動小数点数を利用していた。IEEE 754規格によると単精度浮動小数点数は

- 仮数部: 23ビット
- 指数部: 8ビット
- 符号ビット: 1ビット

で表現されるため、約7桁の有効数字を持つ
マンデルブロ集合をズームしていくと、中心座標やピクセルあたりの距離などの値が非常に小さくなり、7桁以上の精度が必要になる場合がある

このような場合、単精度浮動小数点数では精度不足により正確な描画ができなくなる

== 解決策: 摂動法

安易な解決策では倍精度浮動小数点数などより高精度な値を使うことが考えられるが、WGSLでは倍精度小数点数がサポートされていないし、ソフトウェアで倍精度演算を実装すると非常に遅くなってしまう

#pause
そこで摂動法(perturbation method)という手法を用いることで、単精度浮動小数点数のまま高精度な描画を実現することができる

#pause
摂動法においては画面上のある基準点における計算結果がわかっていれば、他の点における計算はその基準点からの摂動(差分)を用いて近似的に求めることができるということを用いる

#pagebreak()
まず基準点$C$における高精度な軌道$Z_n$が事前にCPUで計算されているものとする
$
  Z_(n + 1) = Z_n^2 + C, space.quad
  Z_0 = 0
$
つぎに、描画したいピクセルの座標$c$とその軌道$z_n$を基準点からの摂動として表す
$
  c = C + delta c, space.quad
  z_n = Z_n + delta z_n
$
これをもとの漸化式($z_(n + 1) = z_n^2 + c$)に代入すると
$
  Z_(n + 1) + delta z_(n + 1) &= (Z_n + delta z_n)^2 + C + delta c \
  delta z_(n + 1) &= (Z_n + delta z_n)^2 - Z_(n + 1) + delta c \
  &= 2 Z_n delta z_n + (delta z_n)^2 + delta c
$
ここで$abs(delta z_n) << abs(Z_n)$という条件を仮定すると$(delta z_n)^2$の項は無視できるため
$
  delta z_(n + 1) approx 2 Z_n delta z_n + delta c
$
となる. つまり基準点$C$における軌道$Z_n$がわかっていれば、各ピクセルにおける摂動$delta z_n$を単精度浮動小数点数で近似的に計算できる

この方法を用いることで、単精度浮動小数点数のまま非常に高いズーム率でのマンデルブロ集合の描画が可能になる。摂動法を用いる前は約10^7倍程度のズームが限界だったが、摂動法を用いることで最大約10^30倍程度までズームできるようになった

== 実装上の難点

摂動法において基準点における軌道計算は高精度で行う必要がある。1つの点に対してのみ行えばよいし、Rustには高精度浮動小数点数を提供するクレートがあるためこれを計算することはそこまで困難ではなかった。ちなみに今回は`num-bigfloat`クレートを用いた

#pause
問題はどのようにして基準点の軌道をGPUに渡すかである。さっき紹介した`AsBindGroup`マクロによるマテリアルのバインディングでは、フィールドとして持てるのは基本的に単純な型(整数や浮動小数点数など)に限られるため、配列を直接渡すことができない。ここでBevyあるあるのドキュメント少なすぎ問題が立ちはだかり、かなり苦しめられた。

#pagebreak()
かなり手探りでたどり着いた方法なので本当は正しくはないやり方かもしれないが、以下に実装の内容とその説明を記述する。

```rust
use bevy::render::storage::ShaderStorageBuffer;
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone, Default)]
struct MandelbrotMaterial {
    #[uniform(0)]
    num_iterations: u32,  // マンデルブロ集合の計算に使う反復回数
    #[uniform(0)]
    range: f32,  // 複素平面上の表示範囲の高さ (WGSL側では f32 になる)
    #[uniform(0)]
    aspect_ratio: f32,   // ウィンドウのアスペクト比 (WGSL側では f32 になる)
    #[uniform(0)]
    pixel_size: f32,   // 1ピクセルあたりの複素平面上の距離 (WGSL側では f32 になる)
    #[storage(1, read_only)]
    base_orbit: Handle<ShaderStorageBuffer> // 基準点での軌道を格納するバッファーのid
}
```

```rust
fn update_material(
    params: Res<MandelbrotParams>,
    mandelblot_material_handle: Res<MandelbrotMaterialHandle>,
    mut material_assets: ResMut<Assets<MandelbrotMaterial>>,
    // シェーダーバッファーを保持しているバッファー
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    // Handleを使って、アセットストレージから実際のマテリアルデータを取得（可変）
    let Some(material) = material_assets.get_mut(&mandelblot_material_handle.0) else {
        return;
    };
    material.update_params(&params);
    // Handle<ShaderStorageBuffer>を用いて &mut ShaderStorageBufferを取得
    let Some(buffer) = buffers.get_mut(&material.base_orbit) else {
        return;
    };
    // 基準点での軌道を高精度で計算して、バッファーにセットする
    buffer.set_data(
        iter::once([ZERO; 2]).chain((0..params.num_iterations)
                .scan([ZERO; 2], |z, _| {
                    let x = z[0] * z[0] - z[1] * z[1] + params.center[0];
                    let y = BigFloat::parse("2.0").unwrap() * z[0] * z[1] + params.center[1];
                    *z = [x, y];
                    Some(*z)
                })
            )
            .map(|[x, y]|
                Vec2::new(x.to_f32(), y.to_f32()) // WGSL側では vec2<f32> になる
            )
            .collect::<Vec<_>>()
    )
}
```

```wgsl
struct MaterialData {
    num_iterations: u32,
    range: f32,
    aspect_ratio: f32,
    pixel_size: f32,
};

struct BaseOrbitBuffer {
    data: array<vec2<f32>>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> material: MaterialData;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<storage, read> base_orbit: BaseOrbitBuffer;
```

ここで`MandelbrotMaterial`構造体の`base_orbit`フィールドに`Handle<ShaderStorageBuffer>`型を用いることで、基準点での軌道データを格納するシェーダーストレージバッファーをマテリアルにバインドしている

```wgsl
// マンデルブロ集合の点が発散するかどうかを判定します
// 発散までにかかる反復回数から脱出時の速度を考慮して補正された値を返します
// 摂動法を使用して高精度計算を行います
fn mandelbrot(dc: vec2<f32>) -> f32 {
    var dz = vec2<f32>(0.0);
    var i = 0u;
    var ref_i = 0u;
    for (; i < material.num_iterations; i ++) {
        // Z_i
        var base_z = base_orbit.data[ref_i];
        let z = base_z + dz;
        let radius2 = dot(z, z);
        if radius2 > ESCAPE_RADIUS * ESCAPE_RADIUS {
            let log_zn = log2(radius2) / 2.0;
            return f32(i) + 1.0 - log2(log_zn);
        }
        // Rebasing
        let dradius2 = dot(dz, dz);
        let baseradius2 = dot(base_z, base_z);
        if dradius2 > baseradius2 || baseradius2 > ESCAPE_RADIUS * ESCAPE_RADIUS * 0.25 {
            dz = z;
            ref_i = 0u;
            base_z = vec2(0.0, 0.0);
        }

        // dz_(i+1) = 2 * Z_i * dz_i + (dz_i)^2 + dc
        dz = 2.0 * vec2(
            base_z.x * dz.x - base_z.y * dz.y,
            base_z.x * dz.y + base_z.y * dz.x
        ) + vec2(
            dz.x * dz.x - dz.y * dz.y,
            2.0 * dz.x * dz.y
        ) + dc;
        ref_i = ref_i + 1u;
    }
    return f32(material.num_iterations);
}
```

まだ説明していなかったが、Rebasingという処理もしている。これは摂動$delta z_n$が基準点$Z_n$に比べて大きくなりすぎた場合、または基準点$Z_n$自体が十分大きくなった場合に、現在の点を新たな基準点として摂動をリセットする処理である。これにより摂動が大きくなりすぎるのを防ぎ、また基準点が発散するよりも前に新たな基準点を設定することで、より安定した計算が可能になる。

= おわりに
#pagebreak()
以上でWGSLとBevyを用いたGPU上でのマンデルブロ集合描画の紹介を終わります

#pause
WebGPUを用いることによって、以下のアルゴリズムを参考にさせてもらった記事にあったPython実装では数秒かかっていた描画計算を毎フレーム実行計算できるようになり、インタラクティブな探索が可能になり非常に満足

参考文献
- #link("https://qiita.com/T-STAR/items/91e1975b19d2d4e6d0dc")[Qiita  Pythonでマンデルブロ集合を美しく描画する(基礎編)]
- #link("https://qiita.com/T-STAR/items/2ef76940f181acbc90f8")[Qiita  Pythonでマンデルブロ集合を美しく描画する(摂動論編)]
- #link("https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set#")[Wikipedia  Plotting algorithms for the Mandelbrot set]