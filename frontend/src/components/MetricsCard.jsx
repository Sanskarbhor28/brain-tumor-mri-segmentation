function MetricCard({ title, value, color = "blue" }) {
  const colors = {
    red: "text-red-600",
    green: "text-green-600",
    blue: "text-blue-600",
    purple: "text-purple-600",
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 text-center">
      <p className="text-gray-500 text-lg mb-2">
        {title}
      </p>

      <p className={`text-3xl font-bold ${colors[color]}`}>
        {value}
      </p>
    </div>
  );
}

export default MetricCard;