function ResultCard({ title, image }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-5">

      <h3 className="text-xl font-bold text-center mb-4">
        {title}
      </h3>

      <div className="border-2 border-dashed border-gray-300 rounded-lg overflow-hidden bg-black min-h-[300px] flex items-center justify-center">

        {image ? (
          <img
            src={image}
            alt={title}
            className="w-full h-[320px] object-contain"
          />
        ) : (
          <p className="text-gray-400">
            No Image
          </p>
        )}

      </div>

    </div>
  );
}

export default ResultCard;