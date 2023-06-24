import React, { useState } from "react";
import axios from "axios";

function App() {
  const [selectedFile, setSelectedFile] = useState();
  const [isFilePicked, setIsFilePicked] = useState(false);
  const [colorizedImage, setColorizedImage] = useState();

  const changeHandler = (event) => {
    setSelectedFile(event.target.files[0]);
    setIsFilePicked(true);
  };

  const handleSubmission = () => {
    const formData = new FormData();

    formData.append('file', selectedFile);

    axios.post('http://localhost:5000/predict', formData)
      .then((response) => {
        setColorizedImage('http://localhost:5000/output/' + response.data.image);
      })
      .catch((error) => console.error('Error:', error));
  };

  return (
    <div className="App">
      <input type="file" name="file" onChange={changeHandler} />
      {isFilePicked ? (
        <div>
          <p>Filename: {selectedFile.name}</p>
          <button onClick={handleSubmission}>Submit</button>
        </div>
      ) : (
        <p>Select a file to show details</p>
      )}
      {colorizedImage && <img src={colorizedImage} alt="colorized version" />}
    </div>
  );
}

export default App;
