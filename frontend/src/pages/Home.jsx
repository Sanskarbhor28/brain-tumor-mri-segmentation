import { useState } from "react";
import API from "../services/api";
import ResultCard from "../components/ResultCard";

const BACKEND_URL = "http://127.0.0.1:8000";

const MODEL_NAMES = {
  unet: "UNet",
  residual_unet: "Residual UNet",
  unetplusplus: "UNet++",
};

function Home() {

  const [file, setFile] = useState(null);

  const [loading, setLoading] = useState(false);

  const [comparison, setComparison] = useState(null);


  // ==========================================================
  // ANALYZE MRI
  // ==========================================================

  const handleAnalyze = async () => {

    if (!file) {
      alert("Please select an MRI image.");
      return;
    }

    try {

      setLoading(true);

      setComparison(null);

      const formData = new FormData();

      formData.append("file", file);


      const response = await API.post(
        "/predict/compare",
        formData
      );


      console.log(
        "Comparison result:",
        response.data
      );


      setComparison(response.data);

    } catch (error) {

      console.error(
        "Prediction error:",
        error
      );

      if (error.response) {

        console.error(
          "Server response:",
          error.response.data
        );

      }

      alert(
        "Prediction failed. Check the backend terminal."
      );

    } finally {

      setLoading(false);

    }

  };


  // ==========================================================
  // GET MODEL RESULT
  // ==========================================================

  const getModelResult = (model) => {

    if (
      !comparison ||
      !comparison.models
    ) {
      return null;
    }

    return comparison.models[model];

  };


  // ==========================================================
  // IMAGE URL
  // ==========================================================

  const getImageUrl = (filename) => {

    if (!filename) {
      return null;
    }

    return `${BACKEND_URL}/outputs/${filename}`;

  };


  // ==========================================================
  // DOWNLOAD
  // ==========================================================

  const downloadImage = (
    url,
    filename
  ) => {

    const link =
      document.createElement("a");

    link.href = url;

    link.download = filename;

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);

  };


  const unet =
    getModelResult("unet");

  const residual =
    getModelResult("residual_unet");

  const unetplusplus =
    getModelResult("unetplusplus");


  return (

    <div className="space-y-10">


      {/* =====================================================
          UPLOAD SECTION
      ===================================================== */}

      <div className="bg-white rounded-2xl shadow-lg p-8">

        <h2 className="text-3xl font-bold text-gray-800 mb-8">
          Upload MRI Scan
        </h2>


        <div className="grid md:grid-cols-2 gap-8">


          {/* MRI FILE */}

          <div>

            <label className="block font-semibold text-lg mb-3">
              MRI Image
            </label>

            <input
              type="file"
              accept="image/*"
              onChange={(e) => {

                setFile(
                  e.target.files?.[0] || null
                );

                setComparison(null);

              }}
              className="
                w-full
                border
                border-gray-400
                rounded-lg
                p-3
                cursor-pointer
              "
            />

            {file && (

              <p className="text-sm text-gray-500 mt-2">

                Selected:
                {" "}
                {file.name}

              </p>

            )}

          </div>


          {/* MODEL INFO */}

          <div>

            <label className="block font-semibold text-lg mb-3">
              Models
            </label>

            <div className="
              border
              border-gray-400
              rounded-lg
              p-3
              bg-gray-50
            ">

              <div className="flex flex-wrap gap-2">

                <span className="
                  bg-blue-100
                  text-blue-800
                  px-3
                  py-1
                  rounded-full
                  font-semibold
                ">
                  UNet
                </span>

                <span className="
                  bg-purple-100
                  text-purple-800
                  px-3
                  py-1
                  rounded-full
                  font-semibold
                ">
                  Residual UNet
                </span>

                <span className="
                  bg-green-100
                  text-green-800
                  px-3
                  py-1
                  rounded-full
                  font-semibold
                ">
                  UNet++
                </span>

              </div>

              <p className="text-sm text-gray-500 mt-3">
                All three models will analyze the MRI.
              </p>

            </div>

          </div>

        </div>


        {/* ANALYZE BUTTON */}

        <button
          onClick={handleAnalyze}
          disabled={loading}
          className="
            mt-8
            bg-blue-700
            hover:bg-blue-800
            disabled:bg-gray-400
            text-white
            px-8
            py-3
            rounded-lg
            font-semibold
            text-lg
            transition
          "
        >

          {loading
            ? "Analyzing All Models..."
            : "Analyze MRI"
          }

        </button>

      </div>


      {/* =====================================================
          ORIGINAL MRI
      ===================================================== */}

      {comparison && file && (

        <div>

          <h2 className="text-3xl font-bold mb-6">
            Original MRI
          </h2>

          <div className="
            bg-white
            rounded-2xl
            shadow-lg
            p-6
            max-w-2xl
          ">

            <ResultCard
              title="Original MRI"
              image={
                URL.createObjectURL(file)
              }
            />

            <button
              onClick={() =>
                downloadImage(
                  URL.createObjectURL(file),
                  file.name
                )
              }
              className="
                mt-4
                w-full
                bg-gray-700
                hover:bg-gray-800
                text-white
                py-3
                rounded-lg
                font-semibold
              "
            >
              Download Original
            </button>

          </div>

        </div>

      )}


      {/* =====================================================
          MODEL COMPARISON
      ===================================================== */}

      {comparison && (

        <div>

          <h2 className="text-3xl font-bold mb-6">
            Model Comparison
          </h2>


          <div className="
            grid
            lg:grid-cols-3
            gap-6
          ">


            {/* =================================================
                UNET
            ================================================= */}

            {unet && (

              <div className="
                bg-white
                rounded-2xl
                shadow-lg
                p-5
              ">

                <h3 className="
                  text-2xl
                  font-bold
                  text-center
                  mb-5
                  text-blue-700
                ">
                  UNet
                </h3>


                <ResultCard
                  title="Predicted Mask"
                  image={
                    getImageUrl(
                      unet.mask_file
                    )
                  }
                />


                <button
                  onClick={() =>
                    downloadImage(
                      getImageUrl(
                        unet.mask_file
                      ),
                      unet.mask_file
                    )
                  }
                  className="
                    w-full
                    mt-3
                    bg-blue-600
                    hover:bg-blue-700
                    text-white
                    py-2
                    rounded-lg
                    font-semibold
                  "
                >
                  Download Mask
                </button>


                <ResultCard
                  title="Tumor Overlay"
                  image={
                    getImageUrl(
                      unet.overlay_file
                    )
                  }
                />


                <button
                  onClick={() =>
                    downloadImage(
                      getImageUrl(
                        unet.overlay_file
                      ),
                      unet.overlay_file
                    )
                  }
                  className="
                    w-full
                    mt-3
                    bg-green-600
                    hover:bg-green-700
                    text-white
                    py-2
                    rounded-lg
                    font-semibold
                  "
                >
                  Download Overlay
                </button>


                <div className="mt-5 space-y-3">

                  <Metric
                    title="Tumor Area"
                    value={`${unet.tumor_percentage}%`}
                    color="text-red-600"
                  />

                  <Metric
                    title="Confidence"
                    value={`${(
                      unet.confidence * 100
                    ).toFixed(2)}%`}
                    color="text-green-600"
                  />

                  <Metric
                    title="Inference Time"
                    value={`${unet.inference_time_ms} ms`}
                    color="text-blue-600"
                  />

                </div>

              </div>

            )}


            {/* =================================================
                RESIDUAL UNET
            ================================================= */}

            {residual && (

              <div className="
                bg-white
                rounded-2xl
                shadow-lg
                p-5
              ">

                <h3 className="
                  text-2xl
                  font-bold
                  text-center
                  mb-5
                  text-purple-700
                ">
                  Residual UNet
                </h3>


                <ResultCard
                  title="Predicted Mask"
                  image={
                    getImageUrl(
                      residual.mask_file
                    )
                  }
                />


                <button
                  onClick={() =>
                    downloadImage(
                      getImageUrl(
                        residual.mask_file
                      ),
                      residual.mask_file
                    )
                  }
                  className="
                    w-full
                    mt-3
                    bg-blue-600
                    hover:bg-blue-700
                    text-white
                    py-2
                    rounded-lg
                    font-semibold
                  "
                >
                  Download Mask
                </button>


                <ResultCard
                  title="Tumor Overlay"
                  image={
                    getImageUrl(
                      residual.overlay_file
                    )
                  }
                />


                <button
                  onClick={() =>
                    downloadImage(
                      getImageUrl(
                        residual.overlay_file
                      ),
                      residual.overlay_file
                    )
                  }
                  className="
                    w-full
                    mt-3
                    bg-green-600
                    hover:bg-green-700
                    text-white
                    py-2
                    rounded-lg
                    font-semibold
                  "
                >
                  Download Overlay
                </button>


                <div className="mt-5 space-y-3">

                  <Metric
                    title="Tumor Area"
                    value={`${residual.tumor_percentage}%`}
                    color="text-red-600"
                  />

                  <Metric
                    title="Confidence"
                    value={`${(
                      residual.confidence * 100
                    ).toFixed(2)}%`}
                    color="text-green-600"
                  />

                  <Metric
                    title="Inference Time"
                    value={`${residual.inference_time_ms} ms`}
                    color="text-blue-600"
                  />

                </div>

              </div>

            )}


            {/* =================================================
                UNET++
            ================================================= */}

            {unetplusplus && (

              <div className="
                bg-white
                rounded-2xl
                shadow-lg
                p-5
              ">

                <h3 className="
                  text-2xl
                  font-bold
                  text-center
                  mb-5
                  text-green-700
                ">
                  UNet++
                </h3>


                <ResultCard
                  title="Predicted Mask"
                  image={
                    getImageUrl(
                      unetplusplus.mask_file
                    )
                  }
                />


                <button
                  onClick={() =>
                    downloadImage(
                      getImageUrl(
                        unetplusplus.mask_file
                      ),
                      unetplusplus.mask_file
                    )
                  }
                  className="
                    w-full
                    mt-3
                    bg-blue-600
                    hover:bg-blue-700
                    text-white
                    py-2
                    rounded-lg
                    font-semibold
                  "
                >
                  Download Mask
                </button>


                <ResultCard
                  title="Tumor Overlay"
                  image={
                    getImageUrl(
                      unetplusplus.overlay_file
                    )
                  }
                />


                <button
                  onClick={() =>
                    downloadImage(
                      getImageUrl(
                        unetplusplus.overlay_file
                      ),
                      unetplusplus.overlay_file
                    )
                  }
                  className="
                    w-full
                    mt-3
                    bg-green-600
                    hover:bg-green-700
                    text-white
                    py-2
                    rounded-lg
                    font-semibold
                  "
                >
                  Download Overlay
                </button>


                <div className="mt-5 space-y-3">

                  <Metric
                    title="Tumor Area"
                    value={`${unetplusplus.tumor_percentage}%`}
                    color="text-red-600"
                  />

                  <Metric
                    title="Confidence"
                    value={`${(
                      unetplusplus.confidence * 100
                    ).toFixed(2)}%`}
                    color="text-green-600"
                  />

                  <Metric
                    title="Inference Time"
                    value={`${unetplusplus.inference_time_ms} ms`}
                    color="text-blue-600"
                  />

                </div>

              </div>

            )}

          </div>

        </div>

      )}


      {/* =====================================================
          SUMMARY TABLE
      ===================================================== */}

      {comparison && (

        <div>

          <h2 className="text-3xl font-bold mb-6">
            Performance Summary
          </h2>


          <div className="
            bg-white
            rounded-2xl
            shadow-lg
            overflow-hidden
          ">

            <div className="overflow-x-auto">

              <table className="w-full">

                <thead className="bg-blue-700 text-white">

                  <tr>

                    <th className="p-4 text-left">
                      Model
                    </th>

                    <th className="p-4">
                      Tumor Area
                    </th>

                    <th className="p-4">
                      Confidence
                    </th>

                    <th className="p-4">
                      Inference Time
                    </th>

                  </tr>

                </thead>


                <tbody>

                  {[
                    unet,
                    residual,
                    unetplusplus
                  ]
                    .filter(Boolean)
                    .map((item) => (

                      <tr
                        key={item.model}
                        className="border-b"
                      >

                        <td className="
                          p-4
                          font-bold
                        ">
                          {
                            MODEL_NAMES[
                              item.model
                            ]
                          }
                        </td>

                        <td className="
                          p-4
                          text-center
                          text-red-600
                          font-semibold
                        ">
                          {item.tumor_percentage}%
                        </td>

                        <td className="
                          p-4
                          text-center
                          text-green-600
                          font-semibold
                        ">
                          {(
                            item.confidence * 100
                          ).toFixed(2)}%
                        </td>

                        <td className="
                          p-4
                          text-center
                          text-blue-600
                          font-semibold
                        ">
                          {
                            item.inference_time_ms
                          } ms
                        </td>

                      </tr>

                    ))}

                </tbody>

              </table>

            </div>

          </div>

        </div>

      )}


      {/* =====================================================
          DISCLAIMER
      ===================================================== */}

      <div className="
        bg-yellow-50
        border
        border-yellow-300
        rounded-xl
        p-5
        text-yellow-800
      ">

        <strong>
          Note:
        </strong>

        {" "}
        This system is intended for research and
        educational purposes. The segmentation
        results should not be considered a medical
        diagnosis.

      </div>

    </div>

  );

}


// ============================================================
// METRIC COMPONENT
// ============================================================

function Metric({
  title,
  value,
  color
}) {

  return (

    <div className="
      bg-gray-50
      rounded-lg
      p-4
      text-center
    ">

      <p className="
        text-gray-500
        text-sm
        font-semibold
      ">
        {title}
      </p>

      <p className={`
        text-2xl
        font-bold
        ${color}
      `}>
        {value}
      </p>

    </div>

  );

}


export default Home;