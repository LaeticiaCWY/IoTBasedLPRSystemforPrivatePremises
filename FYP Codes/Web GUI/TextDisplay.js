import React, { useEffect, useState } from "react";
import { database } from "./firebase"; // Import Firebase database

const TextContentDisplay = ({ textPath1, textPath2 }) => {
  const [textContent1, setTextContent1] = useState("");
  const [textContent2, setTextContent2] = useState("");

  useEffect(() => {
    // Set up a listener for the first text URL
    const dbRef1 = database.ref("lpr/OCR_url");
    const onTextUrlChange1 = (snapshot) => {
      const url = snapshot.val();
      console.log("First Image URL updated:", url); // Log the URL when it's updated

      fetch(url)
        .then((response) => {
          if (!response.ok) {
            throw new Error("Network response was not ok");
          }
          return response.text();
        })
        .then((data) => {
          setTextContent1(data);
          console.log("Text Content 1 updated:", data);
        })
        .catch((error) => {
          console.error("Error fetching text content 1:", error);
        });
    };

    dbRef1.on("value", onTextUrlChange1);

    // Set up a listener for the second text URL
    const dbRef2 = database.ref("lpr/Recognized_url");
    const onTextUrlChange2 = (snapshot) => {
      const url = snapshot.val();
      console.log("First Image URL updated:", url); // Log the URL when it's updated

      fetch(url)
        .then((response) => {
          if (!response.ok) {
            throw new Error("Network response was not ok");
          }
          return response.text();
        })
        .then((data) => {
          setTextContent2(data);
          console.log("Text Content 2 updated:", data);
        })
        .catch((error) => {
          console.error("Error fetching text content 2:", error);
        });
    };

    dbRef2.on("value", onTextUrlChange2);

    // Clean up the listeners when the component unmounts to prevent memory leaks
    return () => {
      dbRef1.off("value", onTextUrlChange1);
      dbRef2.off("value", onTextUrlChange2);
    };
  }, [textPath1, textPath2]);

  return (
    <div className="text-content">
      <div
        className="large-text"
        style={{
          textAlign: "center", // Center the text horizontally
          fontSize: "30px", // Change the font size
          fontWeight: "bold", // Set the font weight to bold
          marginTop: "0px" // Add margin at the bottom
        }}
      >
        <p>
          {textContent1}
          <br /> {/* Add a line break */}
          {textContent2}
        </p>
      </div>
      {/* Other elements with different fonts */}
    </div>
  );
};

export default TextContentDisplay;
