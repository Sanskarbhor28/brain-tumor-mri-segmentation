import { BrowserRouter, Routes, Route } from "react-router-dom";

import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Compare from "./pages/Compare";

function App() {
  return (
    <BrowserRouter>

      <Navbar />

      <main className="min-h-screen bg-[#080b10]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">

          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/compare" element={<Compare />} />
          </Routes>

        </div>
      </main>

    </BrowserRouter>
  );
}

export default App;