import { FaBrain } from "react-icons/fa";

function Navbar() {
  return (
    <nav className="sticky top-0 z-50 w-full border-b border-[#212a35] bg-[#10151c]/95 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="h-[72px] flex items-center justify-between">

          {/* Brand */}
          <div className="flex items-center gap-3 min-w-0">
            <FaBrain className="text-[#2ad9c2] text-3xl shrink-0" />

            <div className="min-w-0">
              <h1 className="text-lg sm:text-2xl font-bold text-white font-sans tracking-tight truncate">
                Brain Tumor MRI Segmentation
              </h1>

              <p className="text-[#94a3b8] text-xs sm:text-sm mt-0.5">
                AI Powered Medical Image Analysis
              </p>
            </div>
          </div>

          {/* Right Side */}
          <div className="flex items-center gap-4 sm:gap-6 ml-4">

            {/* Available Models */}
            <div className="hidden sm:block text-right">
              <p className="text-[10px] sm:text-xs text-[#64748b] uppercase tracking-wider">
                Available Models
              </p>

              <p className="text-xs sm:text-sm font-medium text-[#cbd5e1] mt-1">
                UNet • Residual UNet • UNet++
              </p>
            </div>

            {/* System Status */}
            <div className="flex items-center gap-2 px-3 py-2 rounded-full border border-[#2ad9c2]/30 bg-[#2ad9c2]/5">
              
              <span className="relative flex h-2.5 w-2.5">
                <span className="absolute inline-flex h-full w-full rounded-full bg-[#2ad9c2] opacity-60 animate-ping"></span>

                <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-[#2ad9c2]"></span>
              </span>

              <span className="text-[10px] sm:text-xs font-mono font-semibold tracking-wide text-[#2ad9c2]">
                SYSTEM ONLINE
              </span>

            </div>

          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;