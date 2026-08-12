import { useState } from "react";
import API from "./services/api";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const models = [
    {
      key: "unet",
      name: "UNet",
      color: "text-blue-400",
      border: "border-blue-400/30",
      glow: "shadow-blue-500/10",
      badge: "bg-blue-400/10",
    },
    {
      key: "residual_unet",
      name: "Residual UNet",
      color: "text-purple-400",
      border: "border-purple-400/30",
      glow: "shadow-purple-500/10",
      badge: "bg-purple-400/10",
    },
    {
      key: "unetplusplus",
      name: "UNet++",
      color: "text-emerald-400",
      border: "border-emerald-400/30",
      glow: "shadow-emerald-500/10",
      badge: "bg-emerald-400/10",
    },
  ];

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];

    if (!selectedFile) return;

    setFile(selectedFile);
    setResults(null);
    setError("");

    const imageURL = URL.createObjectURL(selectedFile);
    setPreview(imageURL);
  };

  const analyzeMRI = async () => {
    if (!file) {
      setError("Please select an MRI image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await API.post(
        "/predict/compare",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          timeout: 120000,
        }
      );

      if (response.data?.status !== "success") {
        throw new Error(
          response.data?.message || "Prediction failed."
        );
      }

      setResults(response.data);
    } catch (err) {
      console.error("Prediction error:", err);

      const message =
        err.response?.data?.message ||
        err.message ||
        "Prediction failed. Check the backend terminal.";

      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const getOutputURL = (filename) => {
    if (!filename) return "";

    return `${API.defaults.baseURL}/outputs/${encodeURIComponent(
      filename
    )}`;
  };

  return (
    <div className="min-h-screen bg-[#06090e] text-white">

      {/* =====================================================
          BACKGROUND GLOW
          ===================================================== */}

      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
        <div className="absolute top-1/3 -right-40 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-1/3 w-96 h-96 bg-emerald-500/5 rounded-full blur-3xl" />
      </div>

      {/* =====================================================
          HEADER
          ===================================================== */}

      <header className="relative z-10 border-b border-white/10 bg-[#0b1017]/75 backdrop-blur-xl">

        <div className="max-w-7xl mx-auto px-6 py-5">

          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-5">

            {/* Logo / Title */}

            <div className="flex items-center gap-4">

              <div className="w-12 h-12 rounded-2xl
                              bg-emerald-400/10
                              border border-emerald-300/20
                              flex items-center justify-center
                              text-emerald-300
                              shadow-lg shadow-emerald-500/10">

                <span className="text-2xl">
                  🧠
                </span>

              </div>

              <div>

                <h1 className="text-2xl md:text-3xl font-bold tracking-tight">
                  Brain Tumor MRI Segmentation
                </h1>

                <p className="text-sm text-slate-400 mt-1">
                  AI Powered Medical Image Analysis
                </p>

              </div>

            </div>

            {/* Models + Status */}

            <div className="flex items-center gap-6">

              <div className="hidden sm:block text-right">

                <p className="text-xs uppercase tracking-widest text-slate-500">
                  Available Models
                </p>

                <p className="text-sm font-semibold text-slate-300 mt-1">
                  UNet • Residual UNet • UNet++
                </p>

              </div>

              <div className="flex items-center gap-2
                              px-4 py-2
                              rounded-full
                              bg-emerald-400/5
                              border border-emerald-400/25
                              text-emerald-300">

                <span className="w-2.5 h-2.5 rounded-full
                                 bg-emerald-400
                                 shadow-lg shadow-emerald-400/60" />

                <span className="text-xs font-semibold tracking-wider">
                  SYSTEM ONLINE
                </span>

              </div>

            </div>

          </div>

        </div>

      </header>

      {/* =====================================================
          MAIN
          ===================================================== */}

      <main className="relative z-10 max-w-7xl mx-auto px-6 py-10">

        {/* ===================================================
            UPLOAD CARD
            =================================================== */}

        <section className="
          rounded-3xl
          bg-white/[0.045]
          backdrop-blur-2xl
          border border-white/10
          shadow-2xl shadow-black/30
          p-7 md:p-9
        ">

          <h2 className="text-3xl font-bold mb-8">
            Upload MRI Scan
          </h2>

          <div className="grid lg:grid-cols-2 gap-8">

            {/* MRI IMAGE */}

            <div>

              <h3 className="text-lg font-semibold text-slate-200 mb-4">
                MRI Image
              </h3>

              <label className="
                flex items-center gap-4
                min-h-[72px]
                px-4
                rounded-xl
                bg-black/20
                border border-white/10
                hover:border-emerald-400/40
                transition
                cursor-pointer
              ">

                <span className="
                  shrink-0
                  px-5 py-3
                  rounded-lg
                  bg-emerald-400/10
                  border border-emerald-400/30
                  text-emerald-300
                  font-semibold
                  hover:bg-emerald-400/20
                  transition
                ">
                  Choose File
                </span>

                <span className="text-sm text-slate-400 truncate">
                  {file ? file.name : "No file chosen"}
                </span>

                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                />

              </label>

              {file && (
                <p className="text-sm text-slate-500 mt-3">
                  Selected: {file.name}
                </p>
              )}

            </div>

            {/* MODELS */}

            <div>

              <h3 className="text-lg font-semibold text-slate-200 mb-4">
                Models
              </h3>

              <div className="
                rounded-xl
                bg-black/20
                border border-white/10
                p-5
              ">

                <div className="flex flex-wrap gap-3">

                  {models.map((model) => (

                    <span
                      key={model.key}
                      className={`
                        px-4 py-2
                        rounded-full
                        ${model.badge}
                        ${model.color}
                        border ${model.border}
                        text-sm font-semibold
                      `}
                    >
                      {model.name}
                    </span>

                  ))}

                </div>

                <p className="text-sm text-slate-400 mt-4">
                  All three models will analyze the MRI.
                </p>

              </div>

            </div>

          </div>

          {/* ANALYZE BUTTON */}

          <button
            onClick={analyzeMRI}
            disabled={!file || loading}
            className="
              mt-8
              px-8 py-4
              rounded-xl
              bg-gradient-to-r
              from-emerald-500
              to-cyan-500
              hover:from-emerald-400
              hover:to-cyan-400
              disabled:from-slate-600
              disabled:to-slate-600
              disabled:cursor-not-allowed
              text-white
              font-bold
              shadow-lg
              shadow-emerald-500/20
              transition-all
              duration-300
              hover:-translate-y-0.5
            "
          >

            {loading
              ? "Analyzing All Models..."
              : "Analyze MRI"}

          </button>

          {/* ERROR */}

          {error && (

            <div className="
              mt-5
              rounded-xl
              bg-red-500/10
              border border-red-400/25
              px-5 py-4
              text-red-300
            ">

              <strong>Error:</strong> {error}

            </div>

          )}

        </section>

        {/* ===================================================
            ORIGINAL MRI
            =================================================== */}

        {preview && (

          <section className="mt-12">

            <h2 className="text-2xl font-bold mb-6">
              Original MRI
            </h2>

            <div className="
              max-w-2xl
              rounded-3xl
              bg-white/[0.035]
              backdrop-blur-xl
              border border-white/10
              p-5
              shadow-xl
            ">

              <div className="
                rounded-2xl
                bg-black/30
                border border-white/10
                p-4
              ">

                <p className="
                  text-center
                  text-sm
                  text-slate-400
                  mb-4
                ">
                  Original MRI
                </p>

                <img
                  src={preview}
                  alt="Original MRI"
                  className="
                    w-full
                    max-h-[600px]
                    object-contain
                    rounded-xl
                    bg-black
                  "
                />

              </div>

              <a
                href={preview}
                download={file?.name || "original-mri.jpg"}
                className="
                  block
                  text-center
                  mt-4
                  px-5 py-3
                  rounded-xl
                  bg-white/[0.06]
                  border border-white/10
                  text-slate-300
                  hover:bg-white/[0.1]
                  transition
                  font-semibold
                "
              >
                Download Original
              </a>

            </div>

          </section>

        )}

        {/* ===================================================
            MODEL COMPARISON
            =================================================== */}

        {results && (

          <section className="mt-14">

            <div className="flex items-center justify-between mb-7">

              <div>

                <h2 className="text-3xl font-bold">
                  Model Comparison
                </h2>

                <p className="text-slate-500 mt-1">
                  Segmentation results from all three architectures
                </p>

              </div>

            </div>

            <div className="
              grid
              lg:grid-cols-3
              gap-6
            ">

              {models.map((model) => {

                const result =
                  results.models?.[model.key];

                if (!result || result.status !== "success") {

                  return (

                    <div
                      key={model.key}
                      className="
                        rounded-3xl
                        bg-white/[0.035]
                        border border-red-400/20
                        p-6
                      "
                    >

                      <h3 className={`text-xl font-bold ${model.color}`}>
                        {model.name}
                      </h3>

                      <p className="text-red-300 mt-5">
                        Model prediction failed.
                      </p>

                    </div>

                  );

                }

                const maskURL =
                  getOutputURL(result.mask_file);

                const overlayURL =
                  getOutputURL(result.overlay_file);

                return (

                  <div
                    key={model.key}
                    className={`
                      rounded-3xl
                      bg-white/[0.035]
                      backdrop-blur-xl
                      border ${model.border}
                      ${model.glow}
                      shadow-xl
                      p-5
                      transition-all
                      duration-300
                      hover:-translate-y-1
                    `}
                  >

                    {/* MODEL TITLE */}

                    <div className="flex items-center justify-center mb-6">

                      <span className={`
                        w-3 h-3
                        rounded-full
                        mr-3
                        ${model.color.replace(
                          "text-",
                          "bg-"
                        )}
                      `} />

                      <h3 className={`
                        text-xl
                        font-bold
                        ${model.color}
                      `}>
                        {model.name}
                      </h3>

                    </div>

                    {/* MASK */}

                    <div className="
                      rounded-2xl
                      bg-black/30
                      border border-white/10
                      p-4
                    ">

                      <p className="
                        text-center
                        text-sm
                        font-semibold
                        text-slate-400
                        mb-4
                      ">
                        Predicted Mask
                      </p>

                      <div className="
                        aspect-square
                        rounded-xl
                        overflow-hidden
                        bg-black
                        border border-white/10
                      ">

                        <img
                          src={maskURL}
                          alt={`${model.name} predicted mask`}
                          className="
                            w-full
                            h-full
                            object-contain
                          "
                        />

                      </div>

                    </div>

                    {/* DOWNLOAD MASK */}

                    <a
                      href={maskURL}
                      download={result.mask_file}
                      className="
                        block
                        text-center
                        mt-4
                        px-4 py-3
                        rounded-xl
                        bg-white/[0.05]
                        border border-white/10
                        text-slate-300
                        hover:bg-white/[0.1]
                        transition
                        font-semibold
                      "
                    >
                      Download Mask
                    </a>

                    {/* OVERLAY */}

                    <div className="
                      rounded-2xl
                      bg-black/30
                      border border-white/10
                      p-4
                      mt-5
                    ">

                      <p className="
                        text-center
                        text-sm
                        font-semibold
                        text-slate-400
                        mb-4
                      ">
                        Tumor Overlay
                      </p>

                      <div className="
                        rounded-xl
                        overflow-hidden
                        bg-black
                        border border-white/10
                      ">

                        <img
                          src={overlayURL}
                          alt={`${model.name} tumor overlay`}
                          className="
                            w-full
                            object-contain
                          "
                        />

                      </div>

                    </div>

                    {/* DOWNLOAD OVERLAY */}

                    <a
                      href={overlayURL}
                      download={result.overlay_file}
                      className="
                        block
                        text-center
                        mt-4
                        px-4 py-3
                        rounded-xl
                        bg-emerald-400/10
                        border border-emerald-400/20
                        text-emerald-300
                        hover:bg-emerald-400/20
                        transition
                        font-semibold
                      "
                    >
                      Download Overlay
                    </a>

                    {/* METRICS */}

                    <div className="grid gap-3 mt-5">

                      {/* Tumor Area */}

                      <div className="
                        rounded-xl
                        bg-white/[0.04]
                        border border-white/10
                        p-5
                        text-center
                      ">

                        <p className="text-sm text-slate-500">
                          Tumor Area
                        </p>

                        <p className="
                          text-2xl
                          font-bold
                          text-red-400
                          mt-1
                        ">
                          {result.tumor_percentage}%
                        </p>

                      </div>

                      {/* Confidence */}

                      <div className="
                        rounded-xl
                        bg-white/[0.04]
                        border border-white/10
                        p-5
                        text-center
                      ">

                        <p className="text-sm text-slate-500">
                          Confidence
                        </p>

                        <p className="
                          text-2xl
                          font-bold
                          text-emerald-400
                          mt-1
                        ">
                          {(result.confidence * 100).toFixed(2)}%
                        </p>

                      </div>

                      {/* Inference */}

                      <div className="
                        rounded-xl
                        bg-white/[0.04]
                        border border-white/10
                        p-5
                        text-center
                      ">

                        <p className="text-sm text-slate-500">
                          Inference Time
                        </p>

                        <p className="
                          text-2xl
                          font-bold
                          text-blue-400
                          mt-1
                        ">
                          {result.inference_time_ms} ms
                        </p>

                      </div>

                    </div>

                  </div>

                );

              })}

            </div>

          </section>

        )}

        {/* ===================================================
            PERFORMANCE SUMMARY
            =================================================== */}

        {results && (

          <section className="mt-14">

            <h2 className="text-3xl font-bold mb-6">
              Performance Summary
            </h2>

            <div className="
              overflow-x-auto
              rounded-3xl
              bg-white/[0.035]
              backdrop-blur-xl
              border border-white/10
              shadow-xl
            ">

              <table className="w-full text-left">

                <thead>

                  <tr className="
                    border-b
                    border-white/10
                    bg-white/[0.04]
                  ">

                    <th className="px-6 py-5 text-slate-400">
                      Model
                    </th>

                    <th className="px-6 py-5 text-slate-400">
                      Tumor Area
                    </th>

                    <th className="px-6 py-5 text-slate-400">
                      Confidence
                    </th>

                    <th className="px-6 py-5 text-slate-400">
                      Inference Time
                    </th>

                  </tr>

                </thead>

                <tbody>

                  {models.map((model) => {

                    const result =
                      results.models?.[model.key];

                    if (!result || result.status !== "success") {
                      return null;
                    }

                    return (

                      <tr
                        key={model.key}
                        className="
                          border-b
                          border-white/5
                          hover:bg-white/[0.025]
                          transition
                        "
                      >

                        <td className={`
                          px-6 py-5
                          font-semibold
                          ${model.color}
                        `}>
                          {model.name}
                        </td>

                        <td className="
                          px-6 py-5
                          text-red-400
                          font-semibold
                        ">
                          {result.tumor_percentage}%
                        </td>

                        <td className="
                          px-6 py-5
                          text-emerald-400
                          font-semibold
                        ">
                          {(result.confidence * 100).toFixed(2)}%
                        </td>

                        <td className="
                          px-6 py-5
                          text-blue-400
                          font-semibold
                        ">
                          {result.inference_time_ms} ms
                        </td>

                      </tr>

                    );

                  })}

                </tbody>

              </table>

            </div>

          </section>

        )}

        {/* ===================================================
            DISCLAIMER
            =================================================== */}

        <section className="
          mt-12
          rounded-2xl
          bg-yellow-400/[0.07]
          backdrop-blur-xl
          border border-yellow-400/25
          px-6 py-5
          text-yellow-200
        ">

          <p className="text-sm leading-relaxed">

            <strong className="text-yellow-300">
              Note:
            </strong>{" "}

            This system is intended for research and educational
            purposes. The segmentation results should not be
            considered a medical diagnosis.

          </p>

        </section>

      </main>

      {/* =====================================================
          FOOTER
          ===================================================== */}

      <footer className="
        relative z-10
        border-t border-white/10
        mt-16
        py-8
        text-center
        text-sm
        text-slate-600
      ">

        Brain Tumor MRI Segmentation • Research & Educational Project

      </footer>

    </div>
  );
}

export default App;