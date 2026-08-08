import axios from "axios";

const API = axios.create({
  baseURL:
    "https://brain-tumor-mri-segmentation.onrender.com",
});

export default API;