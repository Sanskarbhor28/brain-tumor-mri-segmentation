import { useEffect, useMemo, useState } from "react";
import API from "../services/api";

import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const BACKEND_URL = "http://127.0.0.1:8000";

function Compare() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  // =========================================================
  // MODELS
  // =========================================================

  const models = [
    {
      key: "unet",
      name: "UNet",
      color: "#7c8cf7",
    },
    {
      key: "residual_unet",
      name: "Residual UNet",
      color: "#c98cf0",
    },
    {
      key: "unetplusplus",
      name: "UNet++",
      color: "#33d6a0",
    },
  ];

  // =========================================================
  // FILE CHANGE
  // =========================================================

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) {
      setFile(null);
      setPreviewUrl(null);
      setResult(null);
      return;
    }

    setFile(selectedFile);

    // Important:
    // Selecting a NEW MRI clears the previous analysis.
    setResult(null);

    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
  };

  // =========================================================
  // CLEANUP IMAGE URL
  // =========================================================

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  // =========================================================
  // DOWNLOAD IMAGE
  // =========================================================

  const downloadImage = async (url, filename) => {
    if (!url) {
      alert("Image is not available.");
      return;
    }

    try {
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(
          `Download failed: ${response.status}`
        );
      }

      const blob = await response.blob();

      const blobUrl =
        window.URL.createObjectURL(blob);

      const link =
        document.createElement("a");

      link.href = blobUrl;
      link.download = filename;

      document.body.appendChild(link);

      link.click();

      link.remove();

      window.URL.revokeObjectURL(blobUrl);

    } catch (error) {
      console.error(
        "Download failed:",
        error
      );

      alert(
        "Unable to download the image."
      );
    }
  };

  // =========================================================
  // ANALYZE MRI
  // =========================================================

  const handleAnalyze = async () => {
    if (!file) {
      alert("Please select an MRI image.");
      return;
    }

    try {
      setLoading(true);

      // IMPORTANT:
      // Hide old results while new analysis is running.
      setResult(null);

      const formData = new FormData();

      formData.append("file", file);

      console.log(
        "Sending MRI:",
        file.name
      );

      const response = await API.post(
        "/predict/compare",
        formData
      );

      console.log(
        "COMPARE RESPONSE:",
        response.data
      );

      setResult(response.data);

    } catch (error) {
      console.error(
        "Comparison failed:",
        error
      );

      if (error.response) {
        console.error(
          "Status:",
          error.response.status
        );

        console.error(
          "Backend response:",
          error.response.data
        );
      }

      alert(
        "Model comparison failed. Check the browser console."
      );

    } finally {
      setLoading(false);
    }
  };

  // =========================================================
  // NORMALIZE RESULT
  // =========================================================

  const modelData = useMemo(() => {
    if (!result) {
      return {};
    }

    return result.models ?? result;
  }, [result]);

  // =========================================================
  // GET MODEL RESULT
  // =========================================================

  const getModelResult = (key) => {
    if (!result) {
      return null;
    }

    return modelData?.[key] ?? null;
  };

  // =========================================================
  // OUTPUT URL
  // =========================================================

  const getOutputUrl = (filename) => {
    if (!filename) {
      return null;
    }

    return `${BACKEND_URL}/outputs/${filename}`;
  };

  // =========================================================
  // CHART DATA
  // =========================================================

  const chartData = useMemo(() => {
    if (!result) {
      return [];
    }

    return models.map((model) => {
      const data =
        modelData?.[model.key];

      return {
        model: model.name,

        tumorArea: Number(
          data?.tumor_percentage ?? 0
        ),

        confidence: Number(
          (
            Number(
              data?.confidence ?? 0
            ) * 100
          ).toFixed(2)
        ),

        inferenceTime: Number(
          data?.inference_time_ms ?? 0
        ),

        color: model.color,
      };
    });
  }, [result, modelData]);

  // =========================================================
  // RENDER
  // =========================================================

  return (
    <div className="min-h-screen bg-[#080b10] text-white">

      <main
        className="
          max-w-7xl
          mx-auto
          px-4
          sm:px-6
          lg:px-8
          py-8
        "
      >

        {/* =====================================================
            COMPARE MODELS / UPLOAD
        ===================================================== */}

        <section
          className="
            bg-[#10151c]
            border
            border-[#212a35]
            rounded-[10px]
            p-5
            sm:p-6
          "
        >

          <h2
            className="
              text-2xl
              font-bold
              text-white
              mb-6
            "
          >
            Compare Models
          </h2>


          <div
            className="
              grid
              grid-cols-1
              md:grid-cols-2
              gap-5
            "
          >

            {/* MRI */}

            <div>

              <label
                className="
                  block
                  text-sm
                  font-semibold
                  text-[#cbd5e1]
                  mb-2
                "
              >
                MRI Image
              </label>

              <div
                className="
                  border
                  border-[#212a35]
                  rounded-lg
                  bg-[#0b1016]
                  p-3
                "
              >

                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="
                    w-full
                    text-sm
                    text-[#cbd5e1]

                    file:mr-4
                    file:rounded-md
                    file:border
                    file:border-[#2ad9c2]
                    file:bg-transparent
                    file:px-3
                    file:py-2
                    file:text-[#2ad9c2]
                    file:font-semibold

                    hover:file:bg-[#2ad9c2]/10

                    focus:outline-none
                    focus-visible:ring-2
                    focus-visible:ring-[#2ad9c2]
                  "
                />

              </div>


              {file && (
                <p
                  className="
                    text-xs
                    text-[#64748b]
                    mt-2
                  "
                >
                  Selected: {file.name}
                </p>
              )}

            </div>


            {/* MODELS */}

            <div>

              <label
                className="
                  block
                  text-sm
                  font-semibold
                  text-[#cbd5e1]
                  mb-2
                "
              >
                Models
              </label>

              <div
                className="
                  border
                  border-[#212a35]
                  rounded-lg
                  bg-[#0b1016]
                  p-4
                "
              >

                <div
                  className="
                    flex
                    flex-wrap
                    gap-2
                  "
                >

                  {models.map((model) => (
                    <span
                      key={model.key}
                      className="
                        px-3
                        py-1
                        rounded-full
                        text-xs
                        font-semibold
                        border
                      "
                      style={{
                        color: model.color,
                        borderColor:
                          `${model.color}66`,
                        backgroundColor:
                          `${model.color}12`,
                      }}
                    >
                      {model.name}
                    </span>
                  ))}

                </div>

                <p
                  className="
                    text-xs
                    text-[#64748b]
                    mt-3
                  "
                >
                  All three models will analyze the MRI.
                </p>

              </div>

            </div>

          </div>


          {/* ANALYZE */}

          <button
            type="button"
            onClick={handleAnalyze}
            disabled={loading}
            className="
              mt-6
              px-6
              py-3
              rounded-lg
              font-semibold
              text-[#06100e]

              bg-gradient-to-r
              from-[#2ad9c2]
              to-[#4de4cf]

              hover:brightness-110

              disabled:opacity-50
              disabled:cursor-not-allowed

              transition

              focus-visible:outline-none
              focus-visible:ring-2
              focus-visible:ring-[#2ad9c2]
            "
          >
            {loading
              ? "Analyzing..."
              : "Analyze MRI"}
          </button>

        </section>


        {/* =====================================================
            ORIGINAL MRI
            Shows after selecting an image.
        ===================================================== */}

        {file && previewUrl && (

          <section className="mt-8">

            <h2
              className="
                text-2xl
                font-bold
                text-white
                mb-4
              "
            >
              Original MRI
            </h2>


            <div
              className="
                bg-[#10151c]
                border
                border-[#212a35]
                rounded-[10px]
                p-4
                max-w-3xl
              "
            >

              <div
                className="
                  bg-[#161c25]
                  border
                  border-[#212a35]
                  rounded-lg
                  p-4
                "
              >

                <h3
                  className="
                    text-center
                    text-sm
                    font-semibold
                    text-[#e2e8f0]
                    mb-4
                  "
                >
                  Original MRI
                </h3>


                <div
                  className="
                    bg-black
                    border
                    border-[#212a35]
                    rounded-lg
                    overflow-hidden
                    flex
                    justify-center
                  "
                >

                  <img
                    src={previewUrl}
                    alt="Original MRI"
                    className="
                      max-w-full
                      max-h-[520px]
                      object-contain
                    "
                  />

                </div>

              </div>


              <button
                type="button"
                onClick={() =>
                  downloadImage(
                    previewUrl,
                    file.name ||
                      "original-mri.jpg"
                  )
                }
                className="
                  mt-3
                  block
                  w-full
                  text-center
                  px-4
                  py-3
                  rounded-lg
                  border
                  border-[#212a35]
                  text-[#cbd5e1]

                  hover:border-[#2ad9c2]
                  hover:text-[#2ad9c2]

                  transition

                  focus-visible:outline-none
                  focus-visible:ring-2
                  focus-visible:ring-[#2ad9c2]
                "
              >
                Download Original
              </button>

            </div>

          </section>

        )}


        {/* =====================================================
            LOADING STATE
            Shows ONLY while API request is running.
        ===================================================== */}

        {loading && (

          <section
            className="
              mt-8
              bg-[#10151c]
              border
              border-[#212a35]
              rounded-[10px]
              p-8
              text-center
            "
          >

            <div
              className="
                inline-block
                w-8
                h-8
                border-2
                border-[#212a35]
                border-t-[#2ad9c2]
                rounded-full
                animate-spin
                mb-4
              "
            />

            <p
              className="
                text-[#2ad9c2]
                font-semibold
              "
            >
              Analyzing MRI...
            </p>

            <p
              className="
                text-sm
                text-[#64748b]
                mt-2
              "
            >
              Running UNet, Residual UNet and UNet++.
            </p>

          </section>

        )}


        {/* =====================================================
            EVERYTHING BELOW THIS POINT ONLY APPEARS
            AFTER SUCCESSFUL ANALYSIS
        ===================================================== */}

        {result && !loading && (

          <>

            {/* =================================================
                MODEL COMPARISON
            ================================================= */}

            <section className="mt-8">

              <h2
                className="
                  text-2xl
                  font-bold
                  text-white
                  mb-4
                "
              >
                Model Comparison
              </h2>


              <div
                className="
                  grid
                  grid-cols-1
                  md:grid-cols-3
                  gap-4
                "
              >

                {models.map((model) => {

                  const modelResult =
                    getModelResult(
                      model.key
                    );

                  return (

                    <div
                      key={model.key}
                      className="
                        bg-[#10151c]
                        border
                        border-[#212a35]
                        rounded-[10px]
                        p-4
                        overflow-hidden
                      "
                      style={{
                        borderTop:
                          `3px solid ${model.color}`,
                      }}
                    >

                      {/* MODEL NAME */}

                      <div
                        className="
                          flex
                          items-center
                          justify-center
                          gap-2
                          mb-5
                        "
                      >

                        <span
                          className="
                            w-2.5
                            h-2.5
                            rounded-full
                          "
                          style={{
                            backgroundColor:
                              model.color,
                            boxShadow:
                              `0 0 8px ${model.color}66`,
                          }}
                        />

                        <h3
                          className="
                            text-lg
                            font-bold
                          "
                          style={{
                            color:
                              model.color,
                          }}
                        >
                          {model.name}
                        </h3>

                      </div>


                      {/* MASK */}

                      <div
                        className="
                          bg-[#161c25]
                          border
                          border-[#212a35]
                          rounded-lg
                          p-3
                        "
                      >

                        <h4
                          className="
                            text-sm
                            font-semibold
                            text-center
                            text-[#cbd5e1]
                            mb-3
                          "
                        >
                          Predicted Mask
                        </h4>

                        <div
                          className="
                            bg-black
                            border
                            border-[#212a35]
                            rounded-lg
                            overflow-hidden
                          "
                        >

                          <img
                            src={getOutputUrl(
                              modelResult?.mask_file
                            )}
                            alt={`${model.name} predicted mask`}
                            className="
                              w-full
                              aspect-square
                              object-contain
                            "
                          />

                        </div>

                      </div>


                      {/* DOWNLOAD MASK */}

                      <button
                        type="button"
                        onClick={() =>
                          downloadImage(
                            getOutputUrl(
                              modelResult?.mask_file
                            ),
                            `${model.key}-mask.png`
                          )
                        }
                        className="
                          mt-3
                          w-full
                          px-4
                          py-2.5
                          rounded-lg
                          border
                          font-semibold
                          text-sm
                          transition

                          focus-visible:outline-none
                          focus-visible:ring-2
                          focus-visible:ring-[#2ad9c2]
                        "
                        style={{
                          color:
                            model.color,
                          borderColor:
                            `${model.color}99`,
                        }}
                      >
                        Download Mask
                      </button>


                      {/* OVERLAY */}

                      <div
                        className="
                          bg-[#161c25]
                          border
                          border-[#212a35]
                          rounded-lg
                          p-3
                          mt-4
                        "
                      >

                        <h4
                          className="
                            text-sm
                            font-semibold
                            text-center
                            text-[#cbd5e1]
                            mb-3
                          "
                        >
                          Tumor Overlay
                        </h4>

                        <div
                          className="
                            bg-black
                            border
                            border-[#212a35]
                            rounded-lg
                            overflow-hidden
                          "
                        >

                          <img
                            src={getOutputUrl(
                              modelResult?.overlay_file
                            )}
                            alt={`${model.name} tumor overlay`}
                            className="
                              w-full
                              aspect-square
                              object-contain
                            "
                          />

                        </div>

                      </div>


                      {/* DOWNLOAD OVERLAY */}

                      <button
                        type="button"
                        onClick={() =>
                          downloadImage(
                            getOutputUrl(
                              modelResult?.overlay_file
                            ),
                            `${model.key}-overlay.png`
                          )
                        }
                        className="
                          mt-3
                          w-full
                          px-4
                          py-2.5
                          rounded-lg
                          border
                          font-semibold
                          text-sm
                          transition

                          focus-visible:outline-none
                          focus-visible:ring-2
                          focus-visible:ring-[#2ad9c2]
                        "
                        style={{
                          color:
                            model.color,
                          borderColor:
                            `${model.color}99`,
                        }}
                      >
                        Download Overlay
                      </button>


                      {/* TUMOR AREA */}

                      <div
                        className="
                          bg-[#161c25]
                          border
                          border-[#212a35]
                          rounded-lg
                          p-4
                          mt-4
                          text-center
                        "
                      >

                        <p
                          className="
                            text-xs
                            text-[#64748b]
                          "
                        >
                          Tumor Area
                        </p>

                        <p
                          className="
                            mt-1
                            text-2xl
                            font-bold
                            font-mono
                            text-[#f0a944]
                          "
                        >
                          {Number(
                            modelResult?.tumor_percentage ??
                              0
                          ).toFixed(2)}
                          %
                        </p>

                      </div>


                      {/* CONFIDENCE */}

                      <div
                        className="
                          bg-[#161c25]
                          border
                          border-[#212a35]
                          rounded-lg
                          p-4
                          mt-3
                        "
                      >

                        <div className="text-center">

                          <p
                            className="
                              text-xs
                              text-[#64748b]
                            "
                          >
                            Confidence
                          </p>

                          <p
                            className="
                              mt-1
                              text-2xl
                              font-bold
                              font-mono
                              text-[#2ad9c2]
                            "
                          >
                            {(
                              Number(
                                modelResult?.confidence ??
                                  0
                              ) * 100
                            ).toFixed(2)}
                            %
                          </p>

                        </div>


                        <div
                          className="
                            mt-3
                            h-1
                            bg-[#212a35]
                            rounded-full
                            overflow-hidden
                          "
                        >

                          <div
                            className="
                              h-full
                              rounded-full
                            "
                            style={{
                              width:
                                `${Math.min(
                                  Math.max(
                                    Number(
                                      modelResult?.confidence ??
                                        0
                                    ) * 100,
                                    0
                                  ),
                                  100
                                )}%`,
                              backgroundColor:
                                model.color,
                            }}
                          />

                        </div>

                      </div>


                      {/* INFERENCE TIME */}

                      <div
                        className="
                          bg-[#161c25]
                          border
                          border-[#212a35]
                          rounded-lg
                          p-4
                          mt-3
                          text-center
                        "
                      >

                        <p
                          className="
                            text-xs
                            text-[#64748b]
                          "
                        >
                          Inference Time
                        </p>

                        <p
                          className="
                            mt-1
                            text-xl
                            font-bold
                            font-mono
                            text-white
                          "
                        >
                          {Number(
                            modelResult?.inference_time_ms ??
                              0
                          ).toFixed(2)}
                          {" "}ms
                        </p>

                      </div>

                    </div>

                  );

                })}

              </div>

            </section>


            {/* =================================================
                PERFORMANCE SUMMARY
            ================================================= */}

            <section className="mt-10">

              <h2
                className="
                  text-2xl
                  font-bold
                  text-white
                  mb-4
                "
              >
                Performance Summary
              </h2>


              <div
                className="
                  overflow-x-auto
                  bg-[#10151c]
                  border
                  border-[#212a35]
                  rounded-[10px]
                "
              >

                <table className="w-full text-sm">

                  <thead>

                    <tr
                      className="
                        border-b
                        border-[#212a35]
                      "
                    >

                      <th
                        className="
                          text-left
                          px-5
                          py-4
                          text-xs
                          uppercase
                          tracking-wider
                          text-[#64748b]
                        "
                      >
                        Model
                      </th>

                      <th
                        className="
                          text-center
                          px-5
                          py-4
                          text-xs
                          uppercase
                          tracking-wider
                          text-[#64748b]
                        "
                      >
                        Tumor Area
                      </th>

                      <th
                        className="
                          text-center
                          px-5
                          py-4
                          text-xs
                          uppercase
                          tracking-wider
                          text-[#64748b]
                        "
                      >
                        Confidence
                      </th>

                      <th
                        className="
                          text-center
                          px-5
                          py-4
                          text-xs
                          uppercase
                          tracking-wider
                          text-[#64748b]
                        "
                      >
                        Inference Time
                      </th>

                    </tr>

                  </thead>


                  <tbody>

                    {models.map((model) => {

                      const modelResult =
                        getModelResult(
                          model.key
                        );

                      return (

                        <tr
                          key={model.key}
                          className="
                            border-b
                            border-[#212a35]
                            last:border-0
                          "
                        >

                          <td
                            className="
                              px-5
                              py-4
                              font-semibold
                              font-mono
                            "
                            style={{
                              color:
                                model.color,
                            }}
                          >

                            <span
                              className="
                                inline-flex
                                items-center
                                gap-2
                              "
                            >

                              <span
                                className="
                                  w-2
                                  h-2
                                  rounded-full
                                "
                                style={{
                                  backgroundColor:
                                    model.color,
                                }}
                              />

                              {model.name}

                            </span>

                          </td>


                          <td
                            className="
                              px-5
                              py-4
                              text-center
                              font-mono
                              text-[#f0a944]
                            "
                          >
                            {Number(
                              modelResult?.tumor_percentage ??
                                0
                            ).toFixed(2)}
                            %
                          </td>


                          <td
                            className="
                              px-5
                              py-4
                              text-center
                              font-mono
                              text-[#2ad9c2]
                            "
                          >
                            {(
                              Number(
                                modelResult?.confidence ??
                                  0
                              ) * 100
                            ).toFixed(2)}
                            %
                          </td>


                          <td
                            className="
                              px-5
                              py-4
                              text-center
                              font-mono
                              text-white
                            "
                          >
                            {Number(
                              modelResult?.inference_time_ms ??
                                0
                            ).toFixed(2)}
                            {" "}ms
                          </td>

                        </tr>

                      );

                    })}

                  </tbody>

                </table>

              </div>

            </section>


            {/* =================================================
                CHARTS
            ================================================= */}

            <section className="mt-10">

              <h2
                className="
                  text-2xl
                  font-bold
                  text-white
                  mb-5
                "
              >
                Performance Charts
              </h2>


              <div
                className="
                  grid
                  grid-cols-1
                  lg:grid-cols-2
                  gap-5
                "
              >

                {/* TUMOR AREA */}

                <div
                  className="
                    bg-[#10151c]
                    border
                    border-[#212a35]
                    rounded-[10px]
                    p-5
                  "
                >

                  <h3
                    className="
                      text-center
                      text-sm
                      font-semibold
                      text-[#e2e8f0]
                      mb-5
                    "
                  >
                    Tumor Area Comparison
                  </h3>


                  <div className="h-[320px]">

                    <ResponsiveContainer
                      width="100%"
                      height="100%"
                    >

                      <BarChart
                        data={chartData}
                      >

                        <CartesianGrid
                          stroke="#212a35"
                          strokeDasharray="3 3"
                        />

                        <XAxis
                          dataKey="model"
                          tick={{
                            fill: "#94a3b8",
                            fontSize: 11,
                          }}
                        />

                        <YAxis
                          tick={{
                            fill: "#94a3b8",
                            fontSize: 11,
                          }}
                          label={{
                            value:
                              "Tumor Area (%)",
                            angle: -90,
                            position:
                              "insideLeft",
                            fill:
                              "#94a3b8",
                          }}
                        />

                        <Tooltip
                          contentStyle={{
                            backgroundColor:
                              "#10151c",
                            border:
                              "1px solid #212a35",
                            borderRadius:
                              "8px",
                            color:
                              "#fff",
                          }}
                        />

                        <Bar
                          dataKey="tumorArea"
                          name="Tumor Area"
                        >

                          {chartData.map(
                            (entry, index) => (
                              <Cell
                                key={
                                  `tumor-${index}`
                                }
                                fill={
                                  entry.color
                                }
                              />
                            )
                          )}

                        </Bar>

                      </BarChart>

                    </ResponsiveContainer>

                  </div>

                </div>


                {/* CONFIDENCE */}

                <div
                  className="
                    bg-[#10151c]
                    border
                    border-[#212a35]
                    rounded-[10px]
                    p-5
                  "
                >

                  <h3
                    className="
                      text-center
                      text-sm
                      font-semibold
                      text-[#e2e8f0]
                      mb-5
                    "
                  >
                    Confidence Comparison
                  </h3>


                  <div className="h-[320px]">

                    <ResponsiveContainer
                      width="100%"
                      height="100%"
                    >

                      <BarChart
                        data={chartData}
                      >

                        <CartesianGrid
                          stroke="#212a35"
                          strokeDasharray="3 3"
                        />

                        <XAxis
                          dataKey="model"
                          tick={{
                            fill: "#94a3b8",
                            fontSize: 11,
                          }}
                        />

                        <YAxis
                          domain={[0, 100]}
                          tick={{
                            fill: "#94a3b8",
                            fontSize: 11,
                          }}
                          label={{
                            value:
                              "Confidence (%)",
                            angle: -90,
                            position:
                              "insideLeft",
                            fill:
                              "#94a3b8",
                          }}
                        />

                        <Tooltip
                          contentStyle={{
                            backgroundColor:
                              "#10151c",
                            border:
                              "1px solid #212a35",
                            borderRadius:
                              "8px",
                            color:
                              "#fff",
                          }}
                        />

                        <Bar
                          dataKey="confidence"
                          name="Confidence"
                        >

                          {chartData.map(
                            (entry, index) => (
                              <Cell
                                key={
                                  `confidence-${index}`
                                }
                                fill={
                                  entry.color
                                }
                              />
                            )
                          )}

                        </Bar>

                      </BarChart>

                    </ResponsiveContainer>

                  </div>

                </div>


                {/* INFERENCE TIME */}

                <div
                  className="
                    lg:col-span-2
                    bg-[#10151c]
                    border
                    border-[#212a35]
                    rounded-[10px]
                    p-5
                  "
                >

                  <h3
                    className="
                      text-center
                      text-sm
                      font-semibold
                      text-[#e2e8f0]
                      mb-5
                    "
                  >
                    Inference Time Comparison
                  </h3>


                  <div className="h-[320px]">

                    <ResponsiveContainer
                      width="100%"
                      height="100%"
                    >

                      <BarChart
                        data={chartData}
                      >

                        <CartesianGrid
                          stroke="#212a35"
                          strokeDasharray="3 3"
                        />

                        <XAxis
                          dataKey="model"
                          tick={{
                            fill: "#94a3b8",
                            fontSize: 11,
                          }}
                        />

                        <YAxis
                          tick={{
                            fill: "#94a3b8",
                            fontSize: 11,
                          }}
                          label={{
                            value:
                              "Inference Time (ms)",
                            angle: -90,
                            position:
                              "insideLeft",
                            fill:
                              "#94a3b8",
                          }}
                        />

                        <Tooltip
                          contentStyle={{
                            backgroundColor:
                              "#10151c",
                            border:
                              "1px solid #212a35",
                            borderRadius:
                              "8px",
                            color:
                              "#fff",
                          }}
                        />

                        <Bar
                          dataKey="inferenceTime"
                          name="Inference Time"
                        >

                          {chartData.map(
                            (entry, index) => (
                              <Cell
                                key={
                                  `time-${index}`
                                }
                                fill={
                                  entry.color
                                }
                              />
                            )
                          )}

                        </Bar>

                      </BarChart>

                    </ResponsiveContainer>

                  </div>

                </div>

              </div>

            </section>


            {/* =================================================
                NOTE
            ================================================= */}

            <div
              className="
                mt-8
                border
                border-[#f0a944]/30
                bg-[#f0a944]/5
                rounded-[10px]
                p-4
              "
            >

              <p
                className="
                  text-sm
                  text-[#f0a944]
                "
              >

                <span className="font-bold">
                  Note:
                </span>{" "}

                This system is intended for
                research and educational purposes.
                The segmentation results should not
                be considered a medical diagnosis.

              </p>

            </div>

          </>

        )}

      </main>

    </div>
  );
}

export default Compare;