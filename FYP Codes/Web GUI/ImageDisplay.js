import React, { useEffect, useState } from "react";
import { storage, database } from "./firebase"; // Import Firebase storage and database
import './styles.css';

const ImageDisplay = () => {
  const [imageUrl1, setImageUrl1] = useState(""); // State for the first image URL
  const [imageUrl2, setImageUrl2] = useState(""); // State for the second image URL

  useEffect(() => {
    // Set up listeners to Firebase Realtime Database to listen for changes to the image URLs
    const dbRef1 = database.ref("lpr/YOLO_url"); // Replace with your database path for the first image
    const dbRef2 = database.ref("lpr/WPOD-NET_url"); // Replace with your database path for the second image

    const onImageUrl1Change = (snapshot) => {
      const url = snapshot.val(); // Get the updated image URL from the snapshot
      setImageUrl1(url); // Set the first image URL to state
      console.log("First Image URL updated:", url); // Log the URL when it's updated
    };

    const onImageUrl2Change = (snapshot) => {
      const url = snapshot.val(); // Get the updated image URL from the snapshot
      setImageUrl2(url); // Set the second image URL to state
      console.log("Second Image URL updated:", url); // Log the URL when it's updated
    };

    dbRef1.on("value", onImageUrl1Change);
    dbRef2.on("value", onImageUrl2Change);

    // Clean up the listeners when the component unmounts to prevent memory leaks
    return () => {
      dbRef1.off("value", onImageUrl1Change);
      dbRef2.off("value", onImageUrl2Change);
    };
  }, []);


  return (
    <div>
      <h2 style={{ textAlign: "center", color: "#000", fontSize: "35px" }}>License Plate Recognition System</h2>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          textAlign: "center",
          marginLeft: "10%", // Adjust the left margin for placement
          marginRight: "10%", // Adjust the right margin for placement
        }}
      >
        <div
          style={{
            flex: "1",
            textAlign: "center",
          }}
        >
          {imageUrl1 && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                minHeight: "50vh", // Make the container at least the height of the viewport
                textAlign: "center",
              }}
            >
              <div style={{ textAlign: "center", maxWidth: "100%", maxHeight: "350px", margin: "0 auto" }}>
                <img
                  src={imageUrl1}
                  alt=""
                  key={imageUrl1}
                  style={{
                    display: "block",
                    margin: "0 auto",
                    width: "450px", // Set the fixed width of the container
                    height: "350px", // Set the fixed height of the container
                    objectFit: "cover", // Crop the image to fit within the fixed dimensions
                    border: "2px solid black", // Border style applied to the image
                  }}
                />
              </div>
              <p style={{ margin: "10px 0 0", fontSize: "18px", fontWeight: "bold" }}>Vehicle Image</p>
            </div>
          )}
        </div>
        <div
          style={{
            flex: "1",
            maxWidth: "45%",
            display: "inline-block",
            textAlign: "center",
          }}
        >
          {imageUrl2 && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                minHeight: "70vh", // Center vertically within the viewport
                textAlign: "center",
              }}
            >
              <div style={{ textAlign: "center", maxWidth: "100%", maxHeight: "350px", margin: "0 auto" }}>
                <img
                  src={imageUrl2}
                  alt=""
                  key={imageUrl2}
                  style={{
                    display: "block",
                    margin: "0 auto",
                    width: "400px", // Set the fixed width of the container (adjust as needed)
                    height: "200px", // Set the fixed height of the container (adjust as needed)
                    objectFit: "cover", // Crop the image to fit within the fixed dimensions
                    border: "2px solid black", // Border style applied to the image
                  }}
                />
              </div>
              <p style={{ margin: "10px 0 0", fontSize: "18px", fontWeight: "bold" }}>WPOD-NET Image</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageDisplay;
