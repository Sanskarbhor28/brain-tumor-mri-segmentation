import { useEffect, useState } from "react";
import API from "../services/api";

const BACKEND_URL =
  "https://brain-tumor-mri-segmentation.onrender.com";

function Compare() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  // =========================================================
  // MODEL
  // =========================================================

  const model = {
    key: "unetplusplus",
    name: "UNet++",
    color: "#33d6a0",
  };

  // =========================================================
  // FILE SELECTION
  // =========================================================

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) {
      setFile(null);
      setPreviewUrl(null);
      setResult(null);
      return;
    }

    // Revoke previous preview
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setFile(selectedFile);
    setResult(null);

    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
  };

  // =========================================================
  // CLEANUP
  // =========================================================

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

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
      setResult(null);

      const formData = new FormData();

      formData.append("file", file);

      console.log("Sending MRI:", file.name);

      // =====================================================
      // RENDER BACKEND
      // UNet++ ONLY
      // =====================================================

      const response = await API.post(
        "/predict/",
        formData,
        {
          params: {
            model_name: "unetplusplus",
          },
        }
      );

      console.log(
        "PREDICTION RESPONSE:",
        response.data
      );

      setResult(response.data);

    } catch (error) {
      console.error(
        "Prediction failed:",
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

      if (error.request) {
        console.error(
          "No response received from backend:",
          error.request
        );
      }

      alert(
        "Prediction failed. Check the backend logs."
      );

    } finally {
      setLoading(false);
    }
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
  // SAFE NUMBER
  // =========================================================

  const getNumber = (value, fallback = 0) => {
    const number = Number(value);

    return Number.isFinite(number)
      ? number
      : fallback;
  };

  // =========================================================
  // RESULT VALUES
  // =========================================================

  const tumorPercentage = getNumber(
    result?.tumor_percentage
  );

  const confidence = getNumber(
    result?.confidence
  );

  const inferenceTime = getNumber(
    result?.inference_time_ms
  );

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

        {/* ===================================================
            UPLOAD SECTION
        =================================================== */}

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
            Upload MRI Scan
          </h2>


          <div
            className="
              grid
              grid-cols-1
              md:grid-cols-2
              gap-5
            "
          >

            {/* =================================================
                MRI IMAGE
            ================================================= */}

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


            {/* =================================================
                MODEL
            ================================================= */}

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
                Model
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

                <span
                  className="
                    inline-block
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
                  UNet++
                </span>


                <p
                  className="
                    text-xs
                    text-[#64748b]
                    mt-3
                  "
                >
                  UNet++ will analyze the MRI.
                </p>

              </div>

            </div>

          </div>


          {/* =================================================
              ANALYZE BUTTON
          ================================================= */}

          <button
            type="button"
            onClick={handleAnalyze}
            disabled={loading || !file}
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
              ? "Analyzing UNet++..."
              : "Analyze MRI"
            }

          </button>

        </section>


        {/* ===================================================
            ORIGINAL MRI
        =================================================== */}

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


              <a
                href={previewUrl}
                download={file.name}
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
                "
              >
                Download Original
              </a>

            </div>

          </section>

        )}


        {/* ===================================================
            RESULT
        =================================================== */}

        {result && (

          <section className="mt-8">

            <h2
              className="
                text-2xl
                font-bold
                text-white
                mb-4
              "
            >
              UNet++ Analysis
            </h2>


            <div
              className="
                bg-[#10151c]
                border
                border-[#212a35]
                rounded-[10px]
                p-4
              "
              style={{
                borderTop:
                  `3px solid ${model.color}`,
              }}
            >

              {/* =============================================
                  MODEL NAME
              ============================================= */}

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
                    color: model.color,
                  }}
                >
                  UNet++
                </h3>

              </div>


              {/* =============================================
                  PREDICTED MASK
              ============================================= */}

              {result.mask_file && (

                <>
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
                          result.mask_file
                        )}
                        alt="UNet++ predicted mask"
                        className="
                          w-full
                          aspect-square
                          object-contain
                        "
                      />

                    </div>

                  </div>


                  <a
                    href={getOutputUrl(
                      result.mask_file
                    )}
                    download
                    className="
                      mt-3
                      block
                      w-full
                      text-center
                      px-4
                      py-2.5
                      rounded-lg
                      border
                      font-semibold
                      text-sm
                    "
                    style={{
                      color: model.color,
                      borderColor:
                        `${model.color}99`,
                    }}
                  >
                    Download Mask
                  </a>
                </>

              )}


              {/* =============================================
                  TUMOR OVERLAY
              ============================================= */}

              {result.overlay_file && (

                <>
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
                          result.overlay_file
                        )}
                        alt="UNet++ tumor overlay"
                        className="
                          w-full
                          aspect-square
                          object-contain
                        "
                      />

                    </div>

                  </div>


                  <a
                    href={getOutputUrl(
                      result.overlay_file
                    )}
                    download
                    className="
                      mt-3
                      block
                      w-full
                      text-center
                      px-4
                      py-2.5
                      rounded-lg
                      border
                      font-semibold
                      text-sm
                    "
                    style={{
                      color: model.color,
                      borderColor:
                        `${model.color}99`,
                    }}
                  >
                    Download Overlay
                  </a>
                </>

              )}


              {/* =============================================
                  METRICS
              ============================================= */}

              <div
                className="
                  grid
                  grid-cols-1
                  md:grid-cols-3
                  gap-3
                  mt-4
                "
              >

                {/* TUMOR AREA */}

                <div
                  className="
                    bg-[#161c25]
                    border
                    border-[#212a35]
                    rounded-lg
                    p-4
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
                    {tumorPercentage.toFixed(2)}%
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
                    text-center
                  "
                >

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
                      confidence * 100
                    ).toFixed(2)}%
                  </p>


                  {/* Progress bar */}

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
                              confidence * 100,
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
                      text-2xl
                      font-bold
                      font-mono
                      text-white
                    "
                  >
                    {inferenceTime.toFixed(2)}
                    ms
                  </p>

                </div>

              </div>


              {/* =============================================
                  OTHER BACKEND INFORMATION
              ============================================= */}

              {result.prediction && (

                <div
                  className="
                    bg-[#161c25]
                    border
                    border-[#212a35]
                    rounded-lg
                    p-4
                    mt-4
                  "
                >

                  <p
                    className="
                      text-xs
                      text-[#64748b]
                    "
                  >
                    Prediction
                  </p>


                  <p
                    className="
                      mt-1
                      text-lg
                      font-semibold
                    "
                  >
                    {String(
                      result.prediction
                    )}
                  </p>

                </div>

              )}

            </div>

          </section>

        )}

      </main>

    </div>
  );
}

export default Compare;